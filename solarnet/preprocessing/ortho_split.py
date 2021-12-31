import os
import numpy as np
from numpy.random import randint
import pandas as pd
from collections import defaultdict
from pathlib import Path
import rasterio
from torch._C import Value
from tqdm import tqdm
from PIL import Image
import glob
import cv2
import pathlib

from typing import Tuple

class OrthoSplitter:
    """

    Attributes:
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in data/README.md
    """

    def __init__(self, ortho_folder: Path, data_folder: Path = Path('data/processed')) -> None:
        self.ortho_folder = ortho_folder
        self.data_folder = data_folder

        self.ortho_org_folder = self.ortho_folder / 'org'
        self.ortho_mask_folder = self.ortho_folder / 'mask'

        if not self.ortho_org_folder.exists():
            raise ValueError("org folder is not exist in {}".format(self.ortho_folder))
        if not self.ortho_mask_folder.exists():
            raise ValueError("org folder is not exist in {}".format(self.ortho_folder))

        # setup; make the necessary folders
        self.processed_folder = data_folder
        if not self.processed_folder.exists(): self. processed_folder.mkdir()

        self.solar_panels = self._setup_folder('solar')
        self.empty = self._setup_folder('empty')

    def _setup_folder(self, folder_name: str) -> Path:

        full_folder_path = self.processed_folder / folder_name
        if not full_folder_path.exists(): full_folder_path.mkdir()

        for subfolder in ['org', 'mask']:
            subfolder_name = full_folder_path / subfolder
            if not subfolder_name.exists(): subfolder_name.mkdir()

        return full_folder_path

    @staticmethod
    def size_okay(image: np.array, imsize: int) -> bool:
        if image.shape == (3, imsize, imsize):
            return True
        return False

    def _split_ortho(self, ortho_file: str, mask_dir: Path, org_dir: Path, ortho_split_size: int, imsize: int):
        import torch

        tmp_file = pathlib.Path(ortho_file)
        mask_file = tmp_file.parents[1] / "mask" / tmp_file.name # マスクのファイル
        
        img = Image.open(ortho_file)
        
        mask_img = Image.open(mask_file).convert('L')
        original = np.expand_dims(np.asarray(img).transpose([2, 1, 0]), axis=0)
        
        mask = np.array(mask_img)

        x = torch.as_tensor(original, dtype=torch.float32)
        x_mask = torch.as_tensor(mask, dtype=torch.float32)

        #patches = x.unfold(2, 224, 224).unfold(3, 224, 224)
        patches = x.unfold(2, ortho_split_size, ortho_split_size).unfold(3, ortho_split_size, ortho_split_size)
        patches_mask = x_mask.unfold(0, ortho_split_size, ortho_split_size).unfold(1, ortho_split_size, ortho_split_size)

        cnt = 1
        size_ratio = imsize / ortho_split_size
        for i in range(patches.shape[2]):
            for j in range(patches.shape[3]):
                cropped = patches[0, : , i, j, :, :].detach().numpy().copy()
                cropped = cropped.transpose([2, 1, 0])
                cropped = cv2.resize(cropped, dsize=None, fx=size_ratio, fy=size_ratio)
                cropped = cropped.transpose([2, 1, 0])
                # gray_img = cv2.cvtColor(cropped.transpose([2, 1, 0]), cv2.COLOR_RGB2GRAY).astype(np.uint8)
                # print(gray_img.shape)
                # gray_img = cv2.resize(gray_img, dsize=None, fx=size_ratio, fy=size_ratio)
                cropped_mask = patches_mask[i, j, :, :].detach().numpy().copy()
                cropped_mask = cv2.resize(cropped_mask, dsize=None, fx=size_ratio, fy=size_ratio)
                # mask_arr = np.zeros_like(gray_img).astype(np.float64)

                arr_name = os.path.split(ortho_file)[1] + '_' + '{:04}'.format(cnt)
                # cv2.imwrite("cropped.png", cropped.transpose())
                # cv2.imwrite("cropped_mask.png", cropped_mask)
                # exit(0)

                if self.size_okay(cropped, imsize):
                    np.save(str(org_dir) + '/' + arr_name, cropped)
                    np.save(str(mask_dir) + '/' + arr_name, cropped_mask)
                    # np.save(str(org_dir) + '/' + arr_name, np.concatenate([np.expand_dims(gray_img, 0), np.expand_dims(gray_img, 0), np.expand_dims(gray_img, 0)]))
                    cnt += 1
        return cnt-1

    def _split_ortho_prediction(self, ortho_file: str, org_dir: Path, ortho_split_size: int, imsize: int):
        import torch
        img = Image.open(ortho_file)
        original = np.expand_dims(np.asarray(img).transpose([2, 0, 1]), axis=0)

        x = torch.as_tensor(original, dtype=torch.float32)

        patches = x.unfold(2, ortho_split_size, ortho_split_size).unfold(3, ortho_split_size, ortho_split_size)

        cnt = 1
        size_ratio = imsize / ortho_split_size
        for i in range(patches.shape[2]):
            for j in range(patches.shape[3]):
                cropped = patches[0, : , i, j, :, :].detach().numpy().copy()
                cropped = cropped.transpose([2, 1, 0])
                cropped = cv2.resize(cropped, dsize=None, fx=size_ratio, fy=size_ratio)

                arr_name = os.path.split(ortho_file)[1] + '_' + '{:04}'.format(cnt)
                print(arr_name)
                cropped = cropped.transpose([2, 1, 0])

                if self.size_okay(cropped, imsize):
                    np.save(str(org_dir) + '/' + arr_name, cropped)
                    cnt += 1
        return cnt-1

    @staticmethod
    def size_okay(image: np.array, imsize: int) -> bool:
        if image.shape == (3, imsize, imsize):
            return True
        return False

    def process(self, ortho_split_size: int, imsize: int=224, use_prediction=False) -> None:
        """Creates the solar and empty images, and their corresponding masks

        Parameters
        ----------
        imsize: int, default: 224
            The size of the images to be generated
        empty_ratio: int, default: 2
            The ratio of images without solar panels to images with solar panels.
            Because images without solar panels are randomly sampled with limited
            patience, having this number slightly > 1 yields a roughly 1:1 ratio.

        Images and masks are saved in {solar, empty}/{org_mask}, with their original
        city in their filename (i.e. {city}_{idx}.npy) where idx is incremented with every
        image-mask combination saved to ensure uniqueness
        """
        solar_dir = self.processed_folder / 'solar'
        solar_mask_dir = solar_dir / 'mask'
        solar_org_dir = solar_dir / 'org'

        ortho_files = glob.glob(str(self.ortho_org_folder) + "/*.tif")
        cnt = 0
        for of in ortho_files:
            if use_prediction:
                cnt += self._split_ortho_prediction(of, solar_org_dir, ortho_split_size, imsize)
            else:
                cnt += self._split_ortho(of, solar_mask_dir, solar_org_dir, ortho_split_size, imsize)

        print(f"Generated {cnt} samples")
