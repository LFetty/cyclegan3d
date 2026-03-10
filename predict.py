import argparse
import numpy as np
import torch
import nibabel as nib
import nibabel.orientations as nibo
from monai.inferers import sliding_window_inference
from monai.transforms import ScaleIntensityRangePercentiles, ScaleIntensityRange
from pathlib import Path
from models import ResUNetGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="3D CycleGAN inference")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--input", required=True, help="Input NIfTI file")
    parser.add_argument("--output", required=True, help="Output NIfTI file")
    parser.add_argument("--direction", default="cbct2ct", choices=["cbct2ct", "ct2cbct"])
    parser.add_argument("--patch_size", type=int, nargs=3, default=[256, 256, 64])
    parser.add_argument("--overlap", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)

    def strip_compiled(sd):
        """Strip _orig_mod. prefix added by torch.compile."""
        if any(k.startswith("_orig_mod.") for k in sd):
            return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        return sd

    G_AB = ResUNetGenerator().to(device)
    G_BA = ResUNetGenerator().to(device)
    G_AB.load_state_dict(strip_compiled(ckpt["G_AB"]))
    G_BA.load_state_dict(strip_compiled(ckpt["G_BA"]))
    model = G_AB if args.direction == "cbct2ct" else G_BA
    model.eval()

    # Load original, remember orientation, reorient to RAS for inference
    orig_nib = nib.load(args.input)
    orig_ornt = nibo.io_orientation(orig_nib.affine)
    ras_nib = nib.as_closest_canonical(orig_nib)

    data = ras_nib.get_fdata(dtype=np.float32)
    img_tensor = torch.from_numpy(data).unsqueeze(0)  # (1, H, W, D)

    if args.direction == "cbct2ct":
        normalize = ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=-1, b_max=1, clip=True)
    else:
        normalize = ScaleIntensityRange(a_min=-1024, a_max=3071, b_min=-1, b_max=1, clip=True)
    img_tensor = normalize(img_tensor).unsqueeze(0).to(device)  # (1, 1, H, W, D)

    patch_size = tuple(args.patch_size)
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=img_tensor,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=model,
            overlap=args.overlap,
            mode="gaussian",
        )

    output_np = output[0, 0].cpu().float().numpy()

    if args.direction == "cbct2ct":
        # [-1, 1] -> HU [-1024, 3071]
        output_np = output_np * ((3071 - (-1024)) / 2) + ((3071 + (-1024)) / 2)

    # Reorient output from RAS back to original orientation
    ras_ornt = nibo.axcodes2ornt(("R", "A", "S"))
    transform = nibo.ornt_transform(ras_ornt, orig_ornt)
    output_orig = nibo.apply_orientation(output_np, transform)

    out_nib = nib.Nifti1Image(output_orig, orig_nib.affine, orig_nib.header)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_nib, args.output)
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
