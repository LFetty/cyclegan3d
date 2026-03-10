"""RegGAN-3D training — pix2pix + registration network (NC+R mode).

    fake_CT  = G(real_CBCT)
    flow     = R(fake_CT, real_CT)
    warped   = SpatialTransform(fake_CT, flow)
    L_G = lambda_adv    * MSE(D(fake_CT), 1)
        + lambda_corr   * L1(warped, real_CT)
        + lambda_smooth * smoothing_loss(flow)

G and R are updated jointly; D is updated separately.
"""
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

from models import (
    ResUNetGenerator, PatchDiscriminator3D, ImagePool,
    RegistrationNet, SpatialTransformer3D, smoothing_loss_3d,
)
from dataset import CycleGANDataset


def parse_args():
    parser = argparse.ArgumentParser(description="3D RegGAN training (CBCT→CT)")
    parser.add_argument("--data_dir", default="/mnt/f/lukas-backup/nnUNet_raw/Dataset020_CBCT_Zimmermann_translation_real/imagesTr/")
    parser.add_argument("--output_dir", default="./outputs_reggan")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--decay_epoch", type=int, default=500)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_D", type=float, default=None)
    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--lambda_corr", type=float, default=20.0)
    parser.add_argument("--lambda_smooth", type=float, default=10.0)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[256, 256, 64])
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--wandb_project", default="reggan3d")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--log_images_every", type=int, default=5)
    parser.add_argument("--no_compile", action="store_true")
    return parser.parse_args()


def make_lr_lambda(decay_epoch, total_epochs):
    def lr_lambda(n):
        if n < decay_epoch:
            return 1.0
        return max(0.0, 1.0 - (n - decay_epoch) / (total_epochs - decay_epoch))
    return lr_lambda


def _state_dict(model):
    return model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()


def save_checkpoint(epoch, G, D, R, opt_G, opt_D, sched_G, sched_D,
                    scaler, wandb_run_id, output_dir):
    ckpt = {
        "epoch": epoch,
        "G": _state_dict(G),
        "D": _state_dict(D),
        "R": _state_dict(R),
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


def log_image_grid(val_batch, G, R, spatial_transform, device, epoch):
    # Use uncompiled modules for eval inference to avoid torch.compile recompilation
    G_mod = G._orig_mod if hasattr(G, "_orig_mod") else G
    R_mod = R._orig_mod if hasattr(R, "_orig_mod") else R
    G_mod.eval(); R_mod.eval()
    with torch.no_grad():
        cbct = val_batch["cbct"].to(device)
        ct   = val_batch["ct"].to(device)
        fake_ct = G_mod(cbct)
        flow    = R_mod(fake_ct, ct)
        warped  = spatial_transform(fake_ct, flow)

    def slices(vol):
        arr = vol[0, 0].cpu().float().numpy()
        r, a, s = arr.shape[0] // 2, arr.shape[1] // 2, arr.shape[2] // 2
        return [(x + 1) / 2 for x in (arr[:, :, s].T, arr[:, a, :].T, arr[r, :, :].T)]

    vols   = [cbct, fake_ct, warped, ct]
    labels = ["Real CBCT", "Fake CT", "Warped CT", "Real CT"]
    planes = ["Axial", "Coronal", "Sagittal"]

    for plane_idx, plane_name in enumerate(planes):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, vol, label in zip(axes, vols, labels):
            ax.imshow(slices(vol)[plane_idx], cmap="gray", vmin=0, vmax=1)
            ax.set_title(label)
            ax.axis("off")
        plt.suptitle(f"Epoch {epoch} — {plane_name}")
        plt.tight_layout()
        wandb.log({f"image_{plane_name.lower()}": wandb.Image(fig)}, step=epoch)
        plt.close(fig)

    G_mod.train(); R_mod.train()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    patch_size = tuple(args.patch_size)

    train_ds = CycleGANDataset(args.data_dir, split="train", patch_size=patch_size, cache_dir=args.cache_dir)
    val_ds   = CycleGANDataset(args.data_dir, split="val",   patch_size=patch_size, cache_dir=args.cache_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} cases, Val: {len(val_ds)} cases")

    G = ResUNetGenerator().to(device)
    D = PatchDiscriminator3D().to(device)
    R = RegistrationNet().to(device)
    spatial_transform = SpatialTransformer3D(patch_size).to(device)

    lr_D = args.lr_D if args.lr_D is not None else args.lr
    opt_G = torch.optim.Adam(
        list(G.parameters()) + list(R.parameters()), lr=args.lr, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))

    lr_lambda = make_lr_lambda(args.decay_epoch, args.epochs)
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    criterion_GAN = nn.MSELoss()
    criterion_reg = nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    pool_fake_ct = ImagePool(pool_size=0)

    start_epoch = 1
    wandb_run_id = None
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        R.load_state_dict(ckpt["R"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        sched_G.load_state_dict(ckpt["sched_G"])
        sched_D.load_state_dict(ckpt["sched_D"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        wandb_run_id = ckpt.get("wandb_run_id")
        print(f"Resumed at epoch {start_epoch}")

    run_name = args.wandb_run_name or f"reggan3d_{int(time.time())}"
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
        torch._functorch.config.donated_buffer = False
        G = torch.compile(G)
        D = torch.compile(D)
        R = torch.compile(R)
        print("Done.")

    val_batch_fixed = next(iter(val_loader))

    for epoch in range(start_epoch, args.epochs + 1):
        G.train(); D.train(); R.train()

        epoch_G = epoch_D = epoch_adv = epoch_corr = epoch_smooth = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            real_cbct = batch["cbct"].to(device)
            real_ct   = batch["ct"].to(device)

            autocast = torch.amp.autocast("cuda", enabled=device.type == "cuda")

            # ── G + R update ─────────────────────────────────────────────────
            opt_G.zero_grad()
            with autocast:
                fake_ct = G(real_cbct)
                flow    = R(fake_ct, real_ct)
                warped  = spatial_transform(fake_ct, flow)
                pred    = D(fake_ct)
                loss_adv    = args.lambda_adv    * criterion_GAN(pred, torch.ones_like(pred))
                loss_corr   = args.lambda_corr   * criterion_reg(warped, real_ct)
                loss_smooth = args.lambda_smooth * smoothing_loss_3d(flow)
                loss_G = loss_adv + loss_corr + loss_smooth
            fake_ct_val = fake_ct.detach()
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)

            # ── D update ─────────────────────────────────────────────────────
            opt_D.zero_grad()
            with autocast:
                pred_real = D(real_ct)
                pred_fake = D(pool_fake_ct.query(fake_ct_val))
                loss_D = 0.5 * (
                    criterion_GAN(pred_real, torch.ones_like(pred_real)) +
                    criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                )
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)

            scaler.update()

            epoch_G      += loss_G.item()
            epoch_D      += loss_D.item()
            epoch_adv    += loss_adv.item()
            epoch_corr   += loss_corr.item()
            epoch_smooth += loss_smooth.item()
            pbar.set_postfix(G=f"{loss_G.item():.3f}", D=f"{loss_D.item():.3f}",
                             corr=f"{loss_corr.item():.3f}")

        sched_G.step()
        sched_D.step()

        n = len(train_loader)
        current_lr = opt_G.param_groups[0]["lr"]
        log_dict = {
            "G_loss": epoch_G / n,
            "D_loss": epoch_D / n,
            "adv_loss": epoch_adv / n,
            "corr_loss": epoch_corr / n,
            "smooth_loss": epoch_smooth / n,
            "lr": current_lr,
        }
        wandb.log(log_dict, step=epoch)
        print(f"Epoch {epoch:4d} | lr: {current_lr:.2e} | " +
              " | ".join(f"{k}: {v:.4f}" for k, v in log_dict.items() if k != "lr"))

        if epoch % args.log_images_every == 0:
            log_image_grid(val_batch_fixed, G, R, spatial_transform, device, epoch)

        if epoch % args.save_every == 0:
            save_checkpoint(epoch, G, D, R, opt_G, opt_D,
                            sched_G, sched_D, scaler, wandb_run_id, args.output_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
