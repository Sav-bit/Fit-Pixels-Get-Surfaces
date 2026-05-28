import glob
import json
import math
import os.path as osp
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils


class TorchMRI3D_Dataloader(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file,
        mode="train",
        num_classes=4,
        resolution=None,
        transforms=None,
        config={},
        coords=None,
        skip_pixels=1.0,
        dataset_dir=None,
    ):
        super(TorchMRI3D_Dataloader, self).__init__()
        assert num_classes == 4 or num_classes == 24 or num_classes == 35, (
            "Only 4 or 24 classes are supported"
        )
        self.json_file = json_file
        self.mode = mode
        self.coords = coords
        self.config = config
        self.resolution = resolution
        self.transforms = transforms
        self.num_classes = num_classes
        self.skip_pixels = skip_pixels
        self.dataset_dir = dataset_dir
        self.build()

    def build(self):
        with open(self.json_file, "r") as f:
            _data = json.load(f)

        self.data = _data[self.mode]
        if self.config.get("N_samples", None) is not None:
            self.data = self.data[: self.config["N_samples"]]
        self.length = len(self.data)

    def __len__(self):
        return self.length

    # NOT USED, probably used for the 2d version of the dataset
    # def read_nib_file(self, file_path):
    #     img = nib.load(file_path).get_fdata()
    #     if img.shape[-1] == 1:
    #         img = img.squeeze(-1)
    #     return img

    def read_nib_volume(self, file_path: str) -> np.ndarray:
        img = nib.load(file_path)
        return img.get_fdata()

    def read_nib_volume_and_affine(
        self, file_path: str
    ) -> tuple[np.ndarray, np.ndarray]:
        img = nib.load(file_path)
        return img.get_fdata(), img.affine

    # def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
    #     if not self.config.get("normalize", False):
    #         return volume
    #     vmin = float(np.min(volume))
    #     vmax = float(np.max(volume))
    #     if vmax - vmin < 1e-8:
    #         return volume - vmin
    #     return (volume - vmin) / (vmax - vmin)
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        if not self.config.get("normalize", False):
            return volume.astype(np.float32)

        volume = volume.astype(np.float32)
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

        image_support_mask = volume != 0

        if image_support_mask.sum() == 0:
            return np.zeros_like(volume, dtype=np.float32)

        vals = volume[image_support_mask]

        low = np.percentile(vals, 0.5)
        high = np.percentile(vals, 99.5)

        if high - low < 1e-8:
            return np.zeros_like(volume, dtype=np.float32)

        volume = np.clip(volume, low, high)
        volume = (volume - low) / (high - low)

        volume[~image_support_mask] = 0.0

        return volume.astype(np.float32)

    def get_coords(self, h: int, w: int, d: int) -> torch.Tensor:
        xx = torch.linspace(-1, 1, h)
        yy = torch.linspace(-1, 1, w)
        zz = torch.linspace(-1, 1, d)

        coords = torch.meshgrid(xx, yy, zz)
        coords = torch.stack(coords, dim=-1)
        return coords

    def get_coords_from_affine(
        self, h: int, w: int, d: int, affine: np.ndarray
    ) -> torch.Tensor:
        ii = torch.arange(h, dtype=torch.float32)
        jj = torch.arange(w, dtype=torch.float32)
        kk = torch.arange(d, dtype=torch.float32)

        grid_i, grid_j, grid_k = torch.meshgrid(ii, jj, kk, indexing="ij")
        ones = torch.ones_like(grid_i)

        voxel_coords = torch.stack(
            [grid_i, grid_j, grid_k, ones], dim=-1
        )  # (H, W, D, 4)
        affine_t = torch.from_numpy(affine).float()
        world_coords = voxel_coords @ affine_t.T  # (H, W, D, 4)

        return world_coords[..., :3]  # (H, W, D, 3)

    def sample_from_3d(
        self,
        volume: np.ndarray,
        segmentation: np.ndarray,
        coords_mtx: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        volume_sampled = volume[
            :: self.skip_pixels, :: self.skip_pixels, :: self.skip_pixels
        ]
        segmentation_sampled = segmentation[
            :: self.skip_pixels, :: self.skip_pixels, :: self.skip_pixels
        ]
        coords_sampled = coords_mtx[
            :: self.skip_pixels, :: self.skip_pixels, :: self.skip_pixels, ...
        ]

        vs, ss, cs = (
            volume_sampled.reshape(-1, 1),
            segmentation_sampled.reshape(-1, 1),
            coords_sampled.reshape(-1, 3),
        )
        assert vs.shape[0] == ss.shape[0] == cs.shape[0], (
            "Shape mismatch for vs, ss, and cs"
        )
        return vs, ss, cs

    def __getitem__(self, idx: int) -> dict:
        data_dict = self.data[idx]
        vol_path = data_dict["img"]  # key is img. but its actually volume
        seg_path = data_dict[f"seg{self.num_classes}"]

        if self.dataset_dir is not None:
            vol_path = osp.join(self.dataset_dir, vol_path)
            seg_path = osp.join(self.dataset_dir, seg_path)

        volume, affine = self.read_nib_volume_and_affine(vol_path)
        segmentation_integers = self.read_nib_volume(seg_path)

        volume = self.normalize_volume(volume)

        coords_mtx = self.get_coords_from_affine(
            volume.shape[0], volume.shape[1], volume.shape[2], affine
        )
        flat = coords_mtx.reshape(-1, 3)
        c_min = flat.min(dim=0).values  # (3,) min per axis
        c_max = flat.max(dim=0).values  # (3,) max per axis
        coords_mtx = 2.0 * (coords_mtx - c_min) / (c_max - c_min) - 1.0

        h, w, d = volume.shape
        if self.skip_pixels != 1.0:
            spatial_res = (
                math.ceil(h / self.skip_pixels),
                math.ceil(w / self.skip_pixels),
                math.ceil(d / self.skip_pixels),
            )
            volume, segmentation_integers, coords = self.sample_from_3d(
                volume, segmentation_integers, coords_mtx
            )
        else:
            spatial_res = (h, w, d)
            coords = coords_mtx.reshape(-1, 3)
            volume = volume.reshape(-1, 1)
            segmentation_integers = segmentation_integers.reshape(-1, 1)

        seg_integer = segmentation_integers.copy()  # H X W X D
        seg_onehot_labels = F.one_hot(
            torch.from_numpy(seg_integer).long(), num_classes=self.num_classes + 1
        ).float()  # H x W x D x NUM_CLASSES

        data_dict = {
            "img": torch.from_numpy(volume),
            "seg": seg_onehot_labels.squeeze(-2),
            "seg_integer": torch.from_numpy(seg_integer),
            "coords": coords,
            "resolution": torch.tensor(spatial_res, dtype=torch.long),
        }
        return data_dict


class CLFFeature(torch.utils.data.Dataset):
    def __init__(self, path, mode):
        super(CLFFeature, self).__init__()
        self.path = path
        self.mode = mode

        self.all_files = glob.glob(osp.join(self.path, self.mode, "*.pth"))
        self.length = len(self.all_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        st = time.time()
        data = torch.load(self.all_files[idx], weights_only=False)
        et = time.time()
        # print('time to load = ' ,et-st)

        return data


# class TorchMRI3D_DataloaderFast(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         json_file,
#         mode="train",
#         num_classes=4,
#         resolution=None,
#         transforms=None,
#         config={},
#         coords=None,
#         skip_pixels=1.0,
#     ):
#         super(TorchMRI3D_DataloaderFast, self).__init__()
#         assert num_classes == 4 or num_classes == 24 or num_classes == 35, (
#             "Only 4 or 24 classes are supported"
#         )
#         self.json_file = json_file
#         self.mode = mode
#         self.coords = coords
#         self.config = config
#         self.resolution = resolution
#         self.transforms = transforms
#         self.num_classes = num_classes
#         self.skip_pixels = skip_pixels
#         self.build()

#     def build(self):
#         with open(self.json_file, "r") as f:
#             _data = json.load(f)

#         self.data = _data[self.mode]
#         if self.config.get("N_samples", None) is not None:
#             self.data = self.data[: self.config["N_samples"]]
#         self.length = len(self.data)

#     def __len__(self):
#         return self.length

#     def read_nib_file(self, file_path):
#         img = nib.load(file_path).get_fdata()
#         if img.shape[-1] == 1:
#             img = img.squeeze(-1)
#         return img

#     def read_nib_volume(self, file_path):
#         img = nib.load(file_path)
#         # print('img shape = ', img.get_fdata().shape)
#         return img.get_fdata()[:, 16:-16, 12:-12]  # crops

#     def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
#         # fallback normalize for this fast loader — uses self.config if available
#         if not getattr(self, "config", {}).get("normalize", False):
#             return volume
#         vmin = float(np.min(volume))
#         vmax = float(np.max(volume))
#         if vmax - vmin < 1e-8:
#             return volume - vmin
#         return (volume - vmin) / (vmax - vmin)

#     def get_coords(self, h, w, d):
#         xx = torch.linspace(-1, 1, h)
#         yy = torch.linspace(-1, 1, w)
#         zz = torch.linspace(-1, 1, d)

#         coords = torch.meshgrid(xx, yy, zz)
#         coords = torch.stack(coords, dim=-1)
#         return coords

#     def sample_from_3d(self, volume, segmentation):
#         # volume = np.reshape(volume, (-1, 1))
#         # segmentation = np.reshape(segmentation, (-1, 1))
#         # # selection_indices = np.random.choice(np.arange(volume.shape[0]), int(self.sample_fraction * volume.shape[0]), replace=False)
#         # selection_indices = np.arange(volume.shape[0])[::int(1/self.sample_fraction)]
#         # volume = volume[selection_indices,...]
#         # segmentation = segmentation[selection_indices,...]
#         # return volume, segmentation, selection_indices

#         h, w, d = volume.shape

#         coords_mtx = self.get_coords(h, w, d)
#         volume_sampled = volume[
#             :: self.skip_pixels, :: self.skip_pixels, :: self.skip_pixels
#         ]
#         segmentation_sampled = segmentation[
#             :: self.skip_pixels, :: self.skip_pixels, :: self.skip_pixels
#         ]
#         coords_sampled = coords_mtx[
#             :: self.skip_pixels, :: self.skip_pixels, :: self.skip_pixels, ...
#         ]

#         # #random
#         # random_h = np.random.choice(np.arange(h), int(h*self.sample_fraction), replace=False)
#         # random_w = np.random.choice(np.arange(w), int(h*self.sample_fraction), replace=False)
#         # random_d = np.random.choice(np.arange(d), int(h*self.sample_fraction), replace=False)
#         # volume_sampled = volume[random_h, random_w, random_d]
#         # segmentation_sampled = segmentation[random_h, random_w, random_d]
#         # coords_sampled = coords_mtx[random_h, random_w, random_d, ...]

#         # vs, ss, cs = volume_sampled.reshape(-1, 1), segmentation_sampled.reshape(-1, 1), coords_sampled.reshape(-1, 3)
#         # assert vs.shape[0] == ss.shape[0] == cs.shape[0], 'Shape mismatch for vs, ss, and cs'
#         # return vs, ss, cs

#         return volume_sampled, segmentation_sampled, coords_sampled

#     def __getitem__(self, idx):
#         data_dict = self.data[idx]
#         vol_path = data_dict["img"]  # key is img. but its actually volume
#         seg_path = data_dict[f"seg{self.num_classes}"]

#         volume = self.read_nib_volume(vol_path)
#         segmentation_integers = self.read_nib_volume(seg_path)
#         coords = self.coords.clone()

#         volume = self.normalize_volume(volume)
#         # print('coords shape = ', coords.shape, 'volume shape = ', volume.shape, 'seg shape = ', segmentation_integers.shape)
#         if self.skip_pixels != 1.0:
#             volume, segmentation_integers, coords = self.sample_from_3d(
#                 volume, segmentation_integers
#             )
#         else:
#             coords = self.get_coords(
#                 volume.shape[0], volume.shape[1], volume.shape[2]
#             )  # .reshape(-1, 3)
#             volume = volume  # .reshape(-1, 1)
#             segmentation_integers = segmentation_integers  # .reshape(-1, 1)

#         seg_integer = segmentation_integers.copy()  # H X W X D
#         seg_onehot_labels = F.one_hot(
#             torch.from_numpy(seg_integer).long(), num_classes=self.num_classes + 1
#         ).float()  # H x W x D x NUM_CLASSES

#         data_dict = {
#             "img": torch.from_numpy(volume),
#             "seg": seg_onehot_labels,
#             "seg_integer": torch.from_numpy(seg_integer),
#             "coords": coords,
#             "resolution": volume.shape,
#         }
#         return data_dict


# class TorchMRI3D_Dataloader_SR(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         json_file,
#         mode="train",
#         num_classes=4,
#         resolution=None,
#         transforms=None,
#         config={},
#         coords=None,
#         skip_pixels=1.0,
#     ):
#         super(TorchMRI3D_Dataloader_SR, self).__init__()
#         assert num_classes == 4 or num_classes == 24 or num_classes == 35, (
#             "Only 4 or 24 classes are supported"
#         )
#         self.json_file = json_file
#         self.mode = mode
#         self.coords = coords
#         self.config = config
#         self.resolution = resolution
#         self.transforms = transforms
#         self.num_classes = num_classes
#         self.skip_pixels = skip_pixels
#         self.build()

#     def build(self):
#         with open(self.json_file, "r") as f:
#             _data = json.load(f)

#         self.data = _data[self.mode]
#         if self.config.get("N_samples", None) is not None:
#             self.data = self.data[: self.config["N_samples"]]
#         self.length = len(self.data)

#     def __len__(self):
#         return self.length

#     def read_nib_file(self, file_path):
#         img = nib.load(file_path).get_fdata()
#         if img.shape[-1] == 1:
#             img = img.squeeze(-1)
#         return img

#     def read_nib_volume(self, file_path):
#         img = nib.load(file_path)
#         # print('img shape = ', img.get_fdata().shape)
#         return img.get_fdata()[:, 16:-16, 12:-12]  # crops

#     def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
#         if not getattr(self, "config", {}).get("normalize", False):
#             return volume
#         vmin = float(np.min(volume))
#         vmax = float(np.max(volume))
#         if vmax - vmin < 1e-8:
#             return volume - vmin
#         return (volume - vmin) / (vmax - vmin)

#     def get_coords(self, h, w, d):
#         xx = torch.linspace(-1, 1, h)
#         yy = torch.linspace(-1, 1, w)
#         zz = torch.linspace(-1, 1, d)

#         coords = torch.meshgrid(xx, yy, zz)
#         coords = torch.stack(coords, dim=-1)
#         return coords

#     def sample_from_3d(self, volume, segmentation):
#         # volume = np.reshape(volume, (-1, 1))
#         # segmentation = np.reshape(segmentation, (-1, 1))
#         # # selection_indices = np.random.choice(np.arange(volume.shape[0]), int(self.sample_fraction * volume.shape[0]), replace=False)
#         # selection_indices = np.arange(volume.shape[0])[::int(1/self.sample_fraction)]
#         # volume = volume[selection_indices,...]
#         # segmentation = segmentation[selection_indices,...]
#         # return volume, segmentation, selection_indices

#         h, w, d = volume.shape

#         coords_mtx = self.get_coords(h, w, d)
#         volume_sampled = volume[
#             :: self.skip_pixels, :: self.skip_pixels, :: self.skip_pixels
#         ]
#         segmentation_sampled = segmentation[
#             :: self.skip_pixels, :: self.skip_pixels, :: self.skip_pixels
#         ]
#         coords_sampled = coords_mtx[
#             :: self.skip_pixels, :: self.skip_pixels, :: self.skip_pixels, ...
#         ]

#         # #random
#         # random_h = np.random.choice(np.arange(h), int(h*self.sample_fraction), replace=False)
#         # random_w = np.random.choice(np.arange(w), int(h*self.sample_fraction), replace=False)
#         # random_d = np.random.choice(np.arange(d), int(h*self.sample_fraction), replace=False)
#         # volume_sampled = volume[random_h, random_w, random_d]
#         # segmentation_sampled = segmentation[random_h, random_w, random_d]
#         # coords_sampled = coords_mtx[random_h, random_w, random_d, ...]

#         vs, ss, cs = (
#             volume_sampled.reshape(-1, 1),
#             segmentation_sampled.reshape(-1, 1),
#             coords_sampled.reshape(-1, 3),
#         )
#         assert vs.shape[0] == ss.shape[0] == cs.shape[0], (
#             "Shape mismatch for vs, ss, and cs"
#         )
#         return vs, ss, cs

#     def __getitem__(self, idx):
#         data_dict = self.data[idx]
#         vol_path = data_dict["img"]  # key is img. but its actually volume
#         seg_path = data_dict[f"seg{self.num_classes}"]

#         volume = self.read_nib_volume(vol_path)
#         segmentation_integers = self.read_nib_volume(seg_path)
#         coords = self.coords.clone()

#         coords_hr = (
#             self.get_coords(volume.shape[0], volume.shape[1], volume.shape[2])
#             .clone()
#             .reshape(-1, 3)
#         )
#         volume_hr = volume.copy().reshape(-1, 1)
#         segmentation_integers_hr = segmentation_integers.copy().reshape(-1, 1)
#         # print('coords shape = ', coords.shape, 'volume shape = ', volume.shape, 'seg shape = ', segmentation_integers.shape)
#         if self.skip_pixels != 1.0:
#             volume, segmentation_integers, coords = self.sample_from_3d(
#                 volume, segmentation_integers
#             )
#         else:
#             coords = self.get_coords(
#                 volume.shape[0], volume.shape[1], volume.shape[2]
#             ).reshape(-1, 3)
#             volume = volume.reshape(-1, 1)
#             segmentation_integers = segmentation_integers.reshape(-1, 1)

#         # normalize high-res copy if requested
#         if getattr(self, "config", {}).get("normalize", False):
#             # normalize the original high-res volume before reshaping
#             try:
#                 orig_vol = self.read_nib_volume(vol_path)
#                 orig_vol = orig_vol - float(np.min(orig_vol))
#                 rng = float(np.max(orig_vol))
#                 if rng > 1e-8:
#                     orig_vol = orig_vol / rng
#                     volume_hr = orig_vol.copy().reshape(-1, 1)
#             except Exception:
#                 pass

#         seg_integer = segmentation_integers.copy()  # H X W X D
#         seg_onehot_labels = F.one_hot(
#             torch.from_numpy(seg_integer).long(), num_classes=self.num_classes + 1
#         ).float()  # H x W x D x NUM_CLASSES

#         seg_onehot_labels_hr = F.one_hot(
#             torch.from_numpy(segmentation_integers_hr).long(),
#             num_classes=self.num_classes + 1,
#         ).float()  # H x W x D x NUM_CLASSES

#         data_dict = {
#             "img": torch.from_numpy(volume),
#             "seg": seg_onehot_labels.squeeze(-2),
#             "seg_integer": torch.from_numpy(seg_integer),
#             "coords": coords,
#             "resolution": volume.shape,
#             "coords_hr": coords_hr,
#             "img_hr": torch.from_numpy(volume_hr),
#             "seg_hr": seg_onehot_labels_hr.squeeze(-2),
#             "seg_integer_hr": torch.from_numpy(segmentation_integers_hr),
#         }
#         return data_dict
