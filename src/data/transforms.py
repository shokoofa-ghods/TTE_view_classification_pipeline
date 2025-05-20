"""
Transforms module
"""

import random
from typing import Any, Callable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms
from PIL import Image

class ApplySameTransformToAllFrames:
    """
    Class to apply transforms to all frames of a video
    """
    def __init__(self):
        self.to_tensor:Callable[[Any],torch.Tensor] = transforms.ToTensor()
        self.color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.4)
        self.random_crop = transforms.RandomCrop(180)

    def get_random_parameters(self,
                              h_flip_pct:float=0.2,
                              v_flip_pct:float=0.2,
                              noise_pct:float=0.2,
                              color_jitter_pct:float=0.2,
                              rand_crop_pct:float=0.4):
        """
        Generate random parameters

        Args:
            h_flip_pct (float optional default=0.2):
                The percent chance to do a horizontal flip
            v_flip_pct (float optional default=0.2)
                The percent chance to do a vertical flip
            color_jitter_pct (float optional default=0.2):
                The precent chance to do a color jitter
            noise_pct (float optional default=0.2):
                The precent chance to apply noise to the image
            rand_crop_pct (float optional default=0.4):
                The precent chance to randomly crop the image
        """
        # This will generate a new set of random parameters each time it's called
        params = {
            'do_horizontal_flip': random.random() < h_flip_pct,
            'do_vertical_flip': random.random() < v_flip_pct,
            'angle': random.uniform(-10, 10),
            'apply_color_jitter': random.random() < color_jitter_pct,
            'apply_noise': random.random() < noise_pct,
            'random_crop': random.random() < rand_crop_pct
        }
        return params

    def __call__(self, list_of_images) -> list[torch.Tensor]:
        # Convert to PIL Images if needed
        def to_image(data:torch.Tensor | Image.Image | np.ndarray) -> Image.Image:
            if isinstance(data, torch.Tensor):
                return transforms.ToPILImage()(data)
            if isinstance(data, np.ndarray):
                return Image.fromarray(data)
            return data
        list_of_pil_images:list[Image.Image] = [to_image(image)
                                                for image in list_of_images]
        pars = self.get_random_parameters()

        # Apply the same transformations to all frames in this sample
        transformed_images = [self.apply_random_transforms(img, pars) for img in list_of_pil_images]
        return transformed_images

    def apply_random_transforms(self,
                                img:Image.Image | torch.Tensor,
                                params:dict[str, Any],
                                out_size:tuple[int, int]=(288,288)) -> torch.Tensor:
        """
        Apply random transforms

        Args:
            img (PIL Image or Tensor): The image to transform
            params (Dict[str, Any]): A mapping of parameters for transform
                Checked parameters:
                    "angle": float -> The angle in radians to rotate by
                    "apply_color_jitter": bool -> True to jitter colors
                    "apply_noise": bool -> True to add gausian noise
                    "random_crop": bool -> True if the image should be cropped
        Returns:
            The transformed image
        """

        # if params['do_horizontal_flip']:
        #     img = TF.hflip(img)
        # if params['do_vertical_flip']:
        #     img = TF.vflip(img)
        img = TF.rotate(img, params['angle']) # type: ignore
        if params['apply_color_jitter']:
            img = self.color_jitter(img)
        if not isinstance(img, torch.Tensor):
            img_t = self.to_tensor(img)
        else:
            img_t = img
        if params['apply_noise']:
            img_t = AddGaussianNoise()(img_t)
        if params['random_crop']:
            img_t = self.random_crop(img_t)
            img_t = transforms.Resize(out_size)(img_t)
        return img_t

class AddGaussianNoise:
    """Adds Gaussian noise to a tensor."""
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

class CustomCrop(object):
    """Crops the bottom 1/10 of the height and 1/8 of the width of the image."""
    def __call__(self, img):
        w, h = img.size
        new_h = h - h // 15
        top = h // 15
        left = w // 10
        return img.crop((left, top, w, new_h))

class TestTimeAugmentation:
    """
    Time Test Augment (Maybe redundant)
    """
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.color_jitter = transforms.ColorJitter(brightness=0.1,
                                                   contrast=0.4,
                                                   saturation=0,
                                                   hue=0)

    def get_random_parameters(self,
                              h_flip_pct:float=0.2,
                              v_flip_pct:float=0.2,
                              noise_pct:float=0.2,
                              color_jitter_pct:float=0.2):
        """
        Generate random parameters for each test instance

        Args:
            h_flip_pct (float optional default=0.2):
                The percent chance to do a horizontal flip
            v_flip_pct (float optional default=0.2)
                The percent chance to do a vertical flip
            color_jitter_pct (float optional default=0.2):
                The precent chance to do a color jitter
            noise_pct (float optional default=0.2):
                The precent chance to apply noise to the image
        """
        params = {
            'do_horizontal_flip': random.random() < h_flip_pct,
            'do_vertical_flip': random.random() < v_flip_pct,
            'angle': random.uniform(-10, 10),
            'apply_color_jitter': random.random() < color_jitter_pct,
            'apply_noise': random.random() < noise_pct
        }
        return params

    def __call__(self, list_of_images):
        # Convert to PIL Images if needed
        list_of_images = [Image.fromarray(image)
                          if isinstance(image, np.ndarray)
                          else image
                          for image in list_of_images]
        pars = self.get_random_parameters()

        # Apply the same transformations to all frames in this sample
        transformed_images = [self.apply_random_transforms(img, pars) for img in list_of_images]
        return transformed_images

    def apply_random_transforms(self, img, params):
        """
        Apply random transforms

        Args:
            img (PIL Image or Tensor): The image to transform
            params (Dict[str, Any]): A mapping of parameters for transform
                Checked parameters:
                    "angle": float -> The angle in radians to rotate by
                    "apply_color_jitter": bool -> True to jitter colors
                    "apply_noise": bool -> True to add gausian noise
                    "random_crop": bool -> True if the image should be cropped
        Returns:
            The transformed image
        """
        if params['do_horizontal_flip']:
            img = TF.hflip(img)
        if params['do_vertical_flip']:
            img = TF.vflip(img)
        img = TF.rotate(img, params['angle'])
        if params['apply_color_jitter']:
            img = self.color_jitter(img)
        img = self.to_tensor(img)
        if params['apply_noise']:
            img = AddGaussianNoise()(img)

        return img

ALL_DATA_TRANSFORMS = transforms.Compose([
    CustomCrop(),
    # transforms.Resize([128, 171], interpolation=InterpolationMode.BILINEAR),

    # transforms.Resize((224, 224)), #resNext50
    # transforms.Resize((256, 256)), #efficientNet-b0
    transforms.Resize((288, 288)), #efficientNet-b2
    # transforms.Resize((275, 275)),
    # transforms.Resize((299, 299)), # JUST FOR INCEPTION
    # transforms.CenterCrop([112, 112]),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
#     transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    # Lambda(lambda x: x/255),
])
