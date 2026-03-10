from pathlib import Path
from torch.utils.data import Dataset
from monai.data import PersistentDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    SpatialPadd,
    RandSpatialCropd,
    RandFlipd,
)


class CycleGANDataset(Dataset):
    def __init__(self, data_dir, split="train", val_cases=10, patch_size=(128, 128, 64),
                 cache_dir="./cache"):
        data_dir = Path(data_dir)
        cbct_files = sorted(data_dir.glob("*_0000.nii.gz"))
        ct_files = sorted(data_dir.glob("*_0001.nii.gz"))

        cbct_map = {f.name.replace("_0000.nii.gz", ""): f for f in cbct_files}
        ct_map = {f.name.replace("_0001.nii.gz", ""): f for f in ct_files}
        common_keys = sorted(set(cbct_map) & set(ct_map))

        if split == "train":
            keys = common_keys[:-val_cases]
        else:
            keys = common_keys[-val_cases:]

        data = [{"cbct": str(cbct_map[k]), "ct": str(ct_map[k])} for k in keys]

        # Deterministic transforms — cached to disk after first run
        deterministic_transforms = Compose([
            LoadImaged(keys=["cbct", "ct"]),
            EnsureChannelFirstd(keys=["cbct", "ct"]),
            Orientationd(keys=["cbct", "ct"], axcodes="RAS"),
            ScaleIntensityRangePercentilesd(
                keys=["cbct"], lower=1, upper=99, b_min=-1, b_max=1, clip=True
            ),
            ScaleIntensityRanged(
                keys=["ct"], a_min=-1024, a_max=3071, b_min=-1, b_max=1, clip=True
            ),
            SpatialPadd(keys=["cbct", "ct"], spatial_size=patch_size, value=-1),
        ])

        # Random transforms — applied fresh every epoch
        self.random_transforms = Compose([
            RandSpatialCropd(keys=["cbct", "ct"], roi_size=patch_size, random_size=False),
            RandFlipd(keys=["cbct", "ct"], prob=0.5, spatial_axis=0),
        ])

        cache_path = Path(cache_dir) / split
        cache_path.mkdir(parents=True, exist_ok=True)
        self.dataset = PersistentDataset(
            data=data,
            transform=deterministic_transforms,
            cache_dir=str(cache_path),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.random_transforms(self.dataset[idx])
