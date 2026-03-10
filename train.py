import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import ResUNetGenerator, PatchDiscriminator3D, ImagePool
from dataset import CycleGANDataset


def parse_args():
    parser = argparse.ArgumentParser(description="3D CycleGAN training (CBCT <-> CT)")
    parser.add_argument("--data_dir", default="/mnt/f/lukas-backup/nnUNet_raw/Dataset018_SynthRAD_Zimmermann/imagesTr/")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--decay_epoch", type=int, default=500, help="Epoch to start linear LR decay to zero")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_D", type=float, default=None, help="Discriminator LR (default: same as --lr)")
    parser.add_argument("--lambda_cycle", type=float, default=10.0)
    parser.add_argument("--lambda_identity", type=float, default=5.0)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[256, 256, 64])
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--wandb_project", default="cyclegan3d")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--log_images_every", type=int, default=5)
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    return parser.parse_args()


def make_lr_lambda(decay_epoch, total_epochs):
    def lr_lambda(n):  # n = current epoch index (0-based)
        if n < decay_epoch:
            return 1.0
        return max(0.0, 1.0 - (n - decay_epoch) / (total_epochs - decay_epoch))
    return lr_lambda


def _state_dict(model):
    """Return state dict from compiled or uncompiled model."""
    return model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()


def save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, opt_G, opt_D,
                    sched_G, sched_D, scaler, wandb_run_id, output_dir):
    ckpt = {
        "epoch": epoch,
        "G_AB": _state_dict(G_AB),
        "G_BA": _state_dict(G_BA),
        "D_A": _state_dict(D_A),
        "D_B": _state_dict(D_B),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "sched_G": sched_G.state_dict(),
        "sched_D": sched_D.state_dict(),
        "scaler": scaler.state_dict(),
        "wandb_run_id": wandb_run_id,
    }
    path = Path(output_dir) / "checkpoints" / f"epoch_{epoch}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"  Saved checkpoint: {path}")


def log_image_grid(val_batch, G_AB, G_BA, device, epoch):
    G_AB.eval()
    G_BA.eval()
    with torch.no_grad():
        cbct = val_batch["cbct"].to(device)
        ct = val_batch["ct"].to(device)

        fake_ct = G_AB(cbct)
        rec_cbct = G_BA(fake_ct)
        fake_cbct = G_BA(ct)
        rec_ct = G_AB(fake_cbct)

    def slices(vol):
        # vol: (B, C, R, A, S) in RAS
        arr = vol[0, 0].cpu().float().numpy()
        r, a, s = arr.shape[0] // 2, arr.shape[1] // 2, arr.shape[2] // 2
        axial    = arr[:, :, s].T        # (A, R) — view from superior
        coronal  = arr[:, a, :].T        # (S, R) — view from anterior
        sagittal = arr[r, :, :].T        # (S, A) — view from right
        return [(img + 1) / 2 for img in (axial, coronal, sagittal)]

    vols = [cbct, fake_ct, rec_cbct, ct, fake_cbct, rec_ct]
    labels = ["Real CBCT", "Fake CT", "Rec CBCT", "Real CT", "Fake CBCT", "Rec CT"]
    planes = ["Axial", "Coronal", "Sagittal"]

    for plane_idx, plane_name in enumerate(planes):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax, vol, label in zip(axes.flat, vols, labels):
            img = slices(vol)[plane_idx]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(label)
            ax.axis("off")
        plt.suptitle(f"Epoch {epoch} — {plane_name}")
        plt.tight_layout()
        wandb.log({f"image_{plane_name.lower()}": wandb.Image(fig)}, step=epoch)
        plt.close(fig)

    G_AB.train()
    G_BA.train()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    patch_size = tuple(args.patch_size)

    train_ds = CycleGANDataset(args.data_dir, split="train", patch_size=patch_size, cache_dir=args.cache_dir)
    val_ds = CycleGANDataset(args.data_dir, split="val", patch_size=patch_size, cache_dir=args.cache_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} cases, Val: {len(val_ds)} cases")

    G_AB = ResUNetGenerator().to(device)
    G_BA = ResUNetGenerator().to(device)
    D_A = PatchDiscriminator3D().to(device)
    D_B = PatchDiscriminator3D().to(device)

    lr_D = args.lr_D if args.lr_D is not None else args.lr

    opt_G = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()), lr=args.lr, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()), lr=lr_D, betas=(0.5, 0.999)
    )

    lr_lambda = make_lr_lambda(args.decay_epoch, args.epochs)
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    pool_fake_ct = ImagePool(pool_size=0)
    pool_fake_cbct = ImagePool(pool_size=0)

    # Resume from checkpoint
    start_epoch = 1
    wandb_run_id = None
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        G_AB.load_state_dict(ckpt["G_AB"])
        G_BA.load_state_dict(ckpt["G_BA"])
        D_A.load_state_dict(ckpt["D_A"])
        D_B.load_state_dict(ckpt["D_B"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        if "sched_G" in ckpt:
            sched_G.load_state_dict(ckpt["sched_G"])
            sched_D.load_state_dict(ckpt["sched_D"])
        else:
            # old checkpoint — fast-forward schedulers to match resumed epoch
            for _ in range(ckpt["epoch"]):
                sched_G.step()
                sched_D.step()
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        wandb_run_id = ckpt.get("wandb_run_id")
        print(f"Resumed at epoch {start_epoch}")

    # W&B: resume same run if we have its ID, otherwise start fresh
    run_name = args.wandb_run_name or f"cyclegan3d_{int(time.time())}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
        config=vars(args),
    )
    wandb_run_id = wandb.run.id

    if not args.no_compile:
        print("Compiling models with torch.compile...")
        torch._functorch.config.donated_buffer = False  # required for retain_graph=True
        G_AB = torch.compile(G_AB)
        G_BA = torch.compile(G_BA)
        D_A = torch.compile(D_A)
        D_B = torch.compile(D_B)
        print("Done.")

    val_batch_fixed = next(iter(val_loader))

    for epoch in range(start_epoch, args.epochs + 1):
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()

        epoch_G_loss = 0.0
        epoch_D_A_loss = 0.0
        epoch_D_B_loss = 0.0
        epoch_cycle_loss = 0.0
        epoch_identity_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            real_cbct = batch["cbct"].to(device)
            real_ct = batch["ct"].to(device)

            autocast = torch.amp.autocast("cuda", enabled=device.type == "cuda")
            opt_G.zero_grad()

            # ── Phase 1: CBCT → CT → CBCT (single backward, no retain_graph) ─
            with autocast:
                fake_ct = G_AB(real_cbct)
                rec_cbct = G_BA(fake_ct)
                pred = D_A(fake_ct)
                loss_GAN_AB = criterion_GAN(pred, torch.ones_like(pred))
                loss_cycle_A = args.lambda_cycle * criterion_cycle(rec_cbct, real_cbct)
                loss_phase1 = loss_GAN_AB + loss_cycle_A
            fake_ct_val = fake_ct.detach()
            scaler.scale(loss_phase1).backward()

            # ── Phase 2: CT → CBCT → CT (single backward, no retain_graph) ───
            with autocast:
                fake_cbct = G_BA(real_ct)
                rec_ct = G_AB(fake_cbct)
                pred = D_B(fake_cbct)
                loss_GAN_BA = criterion_GAN(pred, torch.ones_like(pred))
                loss_cycle_B = args.lambda_cycle * criterion_cycle(rec_ct, real_ct)
                loss_phase2 = loss_GAN_BA + loss_cycle_B
            fake_cbct_val = fake_cbct.detach()
            scaler.scale(loss_phase2).backward()

            # ── Identity (optional, one generator each) ───────────────────────
            loss_identity = torch.tensor(0.0, device=device)
            if args.lambda_identity > 0:
                with autocast:
                    loss_idt_A = args.lambda_identity * criterion_identity(G_AB(real_ct), real_ct)
                scaler.scale(loss_idt_A).backward()
                with autocast:
                    loss_idt_B = args.lambda_identity * criterion_identity(G_BA(real_cbct), real_cbct)
                scaler.scale(loss_idt_B).backward()
                loss_identity = loss_idt_A + loss_idt_B

            scaler.step(opt_G)

            # ── Discriminator update using saved detached fakes ───────────────
            opt_D.zero_grad()
            with autocast:
                pred_real = D_A(real_ct)
                pred_fake = D_A(pool_fake_ct.query(fake_ct_val))
                loss_D_A = 0.5 * (criterion_GAN(pred_real, torch.ones_like(pred_real)) +
                                   criterion_GAN(pred_fake, torch.zeros_like(pred_fake)))
            scaler.scale(loss_D_A).backward()

            with autocast:
                pred_real = D_B(real_cbct)
                pred_fake = D_B(pool_fake_cbct.query(fake_cbct_val))
                loss_D_B = 0.5 * (criterion_GAN(pred_real, torch.ones_like(pred_real)) +
                                   criterion_GAN(pred_fake, torch.zeros_like(pred_fake)))
            scaler.scale(loss_D_B).backward()
            scaler.step(opt_D)

            scaler.update()

            loss_cycle = loss_cycle_A + loss_cycle_B
            loss_G = loss_phase1 + loss_phase2 + loss_identity
            epoch_G_loss += loss_G.item()
            epoch_D_A_loss += loss_D_A.item()
            epoch_D_B_loss += loss_D_B.item()
            epoch_cycle_loss += loss_cycle.item()
            epoch_identity_loss += loss_identity.item()

            pbar.set_postfix(G=f"{loss_G.item():.3f}", D_A=f"{loss_D_A.item():.3f}",
                             D_B=f"{loss_D_B.item():.3f}")

        # LR decay step
        sched_G.step()
        sched_D.step()

        n = len(train_loader)
        current_lr = opt_G.param_groups[0]["lr"]
        log_dict = {
            "G_loss": epoch_G_loss / n,
            "D_A_loss": epoch_D_A_loss / n,
            "D_B_loss": epoch_D_B_loss / n,
            "cycle_loss": epoch_cycle_loss / n,
            "identity_loss": epoch_identity_loss / n,
            "lr": current_lr,
        }
        wandb.log(log_dict, step=epoch)
        print(f"Epoch {epoch:4d} | lr: {current_lr:.2e} | " +
              " | ".join(f"{k}: {v:.4f}" for k, v in log_dict.items() if k != "lr"))

        if epoch % args.log_images_every == 0:
            log_image_grid(val_batch_fixed, G_AB, G_BA, device, epoch)

        if epoch % args.save_every == 0:
            save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, opt_G, opt_D,
                            sched_G, sched_D, scaler, wandb_run_id, args.output_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
