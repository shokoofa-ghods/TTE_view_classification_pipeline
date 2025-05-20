"""
Custom Dataset module
"""

import glob
import os
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, PILToTensor
from PIL import Image
import time
from utils import log

from data.transforms import ApplySameTransformToAllFrames

VALID_LABELS = {'PLAX': 0,
                'PSAX-ves': 1,
                'PSAX-base': 2,
                'PSAX-mid': 3,
                'PSAX-apical': 4,
                'Apical-2ch': 5,
                'Apical-3ch': 6,
                'Apical-5ch': 7,
                'Apical-4ch': 8,
                'Suprasternal': 9,
                'Subcostal': 10
                }
REQUIRES_AUG = [3, 4, 10, 6, 7] # PSAX-apical, PSAX-mid, suprasternal, apical-3ch, apical 5-ch

class CustomDataset(Dataset):
    """
    Custom dataset implementation
    """
    def __init__(self,
                 paths,
                 labels,
                 transform:Callable[[Image.Image | torch.Tensor], Image.Image | torch.Tensor],
                 original_address:str,
                 frames:int,
                    frame_select:Callable[[torch.Tensor, int], torch.Tensor]=
                        lambda t, m: torch.stack([t[len(t) * i // m] for i in range(m)]),
                 training_transform:Callable[[torch.Tensor],
                                             list[torch.Tensor]]
                                             | None=None,
                 normalize:bool=False):
        """
        frames must be set to the number of frames to use per sample
        frame_select(frames, output_size) MUST return a tensor with 'outputsize' frames
        """
        self.transform = transform
        self.training_transform = training_transform
        self.original_address = original_address
        self.frames = frames
        self.labels_require_aug = [3, 4, 10, 6, 7]
        self.frame_select = frame_select
        if normalize:
            distrib:dict[str, list[str]] = {i:[] for i in VALID_LABELS}
            for i, path in enumerate(paths):
                distrib[labels[i]].append(path)
            max_len = max([len(distrib[l]) for l in VALID_LABELS])
            self.paths:list[str] = []
            self.labels:list[str] = []
            self.augment:list[bool] = []
            for i in range(max_len):
                for label, l_paths in distrib.items():
                    self.paths.append(l_paths[i % len(l_paths)])
                    self.labels.append(label)
                    self.augment.append(i >= len(l_paths))
        else:
            self.paths = paths
            self.labels = labels
            self.augment = [False for _ in self.paths]

    def __len__(self):
        return len(self.paths)

    def preprocessing(self, image_array:torch.Tensor, k:int=100) -> torch.Tensor:
        """
        Preprocess a video (image_array)
        
        Args:
            image_array (Tensor shape:(slices, channels, height, width)):
                The video to process
            k (int optional default=100):
                The max number of steps
        Returns:
            The video with unchanged voxels set to 0
        """
        slices, ch, height, width = image_array.shape

        mask = torch.zeros((ch, height, width), dtype=torch.uint8)
        steps = min(k, slices)
        for i in range(steps - 1):
            mask[image_array[i, :, :, :] != image_array[i + 1, :, :, :]] = 1

        output = image_array * mask
        return output

    def _address(self, index:int):
        """
        Get the address of the index
        """
        return os.path.join(self.original_address, 'Dataset', self.paths[index])

    def _index(self, name:str) -> int:
        """
        Get the index from a path string
        """
        return int(os.path.basename(name).split('_')[-1].split('.')[0])

    def __getitem__(self, index):
        start = time.time()
        filepath = self._address(index)
        label = self.labels[index]
        label = torch.tensor([VALID_LABELS[str(label)]])

        # sort the image in based on the order of the slices saved in the folder
        # imgs_paths = sorted(glob.glob(os.path.join(filepath, '*')),
        #                     key=self._index)
        addresses = glob.glob(os.path.join(filepath, '*'))

        imgs_paths = sorted(addresses,
                            key=self._index)

        images_list = []
        for img_path in imgs_paths:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            # if self.augment[index]:
            #     # Do an augmentation of the image
            #     if isinstance(image, Image.Image):
            #         w, h = image.width, image.height
            #     elif isinstance(image, torch.Tensor):
            #         w, h = image.size(2), image.size(1)
            #     transform = ApplySameTransformToAllFrames()
            #     image = transform.apply_random_transforms(image,
            #                                               transform.get_random_parameters(),
            #                                               out_size=(h, w))
            # if isinstance(image, Image.Image):
            #     image = PILToTensor()(image)
            images_list.append(image)


        # creating a 3D tensor image
        sequence_tensor = torch.stack(images_list)
        # chunks = self.sliding_window(sequence_tensor)

        preprocessed_d = self.preprocessing(sequence_tensor)


        # video augmentation on all slices
        if self.training_transform is not None:
            # and label in self.labels_require_aug:
            pil_images:list[Image.Image] = [ToPILImage()(image)
                                            for image in preprocessed_d] # type: ignore
            preprocessed_d = np.stack(pil_images) # type: ignore
            # train_transform = ApplySameTransformToAllFrames()
            preprocessed_d = self.training_transform(preprocessed_d)
            preprocessed_d = torch.stack(preprocessed_d)
        # Return the image and label as tensors

        # log(f"Load time: {time.time() - start:.4f}s")
        return self.frame_select(preprocessed_d, self.frames), label

