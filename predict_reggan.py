"""RegGAN-3D inference — runs G(CBCT) → sCT via sliding-window inference."""
import argparse
import numpy as np
import torch
import nibabel as nib
import nibabel.orientations as nibo
from monai.inferers import sliding_window_inference
from monai.transforms import ScaleIntensityRangePercentiles
from pathlib import Path

from models import ResUNetGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="3D RegGAN inference (CBCT→CT)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Input CBCT NIfTI file")
    parser.add_argument("--output", required=True, help="Output sCT NIfTI file")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[256, 256, 64])
    parser.add_argument("--overlap", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)

    def strip_compiled(sd):
        if any(k.startswith("_orig_mod.") for k in sd):
            return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        return sd

    G = ResUNetGenerator().to(device)
    G.load_state_dict(strip_compiled(ckpt["G"]))
    G.eval()

    orig_nib = nib.load(args.input)
    orig_ornt = nibo.io_orientation(orig_nib.affine)
    ras_nib = nib.as_closest_canonical(orig_nib)

    data = ras_nib.get_fdata(dtype=np.float32)
    img_tensor = torch.from_numpy(data).unsqueeze(0)  # (1, H, W, D)

    normalize = ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=-1, b_max=1, clip=True)
    img_tensor = normalize(img_tensor).unsqueeze(0).to(device)  # (1, 1, H, W, D)

    with torch.no_grad():
        output = sliding_window_inference(
            inputs=img_tensor,
            roi_size=tuple(args.patch_size),
            sw_batch_size=1,
            predictor=G,
            overlap=args.overlap,
            mode="gaussian",
        )

    output_np = output[0, 0].cpu().float().numpy()
    # [-1, 1] → HU [-1024, 3071]
    output_np = output_np * ((3071 - (-1024)) / 2) + ((3071 + (-1024)) / 2)

    ras_ornt = nibo.axcodes2ornt(("R", "A", "S"))
    transform = nibo.ornt_transform(ras_ornt, orig_ornt)
    output_orig = nibo.apply_orientation(output_np, transform)

    out_nib = nib.Nifti1Image(output_orig, orig_nib.affine, orig_nib.header)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_nib, args.output)
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
