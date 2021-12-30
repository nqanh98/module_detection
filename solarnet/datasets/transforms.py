"""
Image transformations, along with their corresponding
mask transformations (if applicable)
"""

import numpy as np
from typing import Tuple, Optional
import cv2


def no_change(image: np.ndarray,
              mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray,
                                                          Optional[np.ndarray]]:
    if mask is None: return image
    return image, mask


def horizontal_flip(image: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray,
                                                                Optional[np.ndarray]]:
    # input: image[channels, height, width]
    image = image[:, :, ::-1]
    if mask is None: return image

    mask = mask[:, ::-1]
    return image, mask


def vertical_flip(image: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray,
                                                              Optional[np.ndarray]]:
    # input: image[channels, height, width]
    image = image[:, ::-1, :]
    if mask is None: return image

    mask = mask[::-1, :]
    return image, mask

def rotation(image: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray,
                                                              Optional[np.ndarray]]:
    # input: image[channels, height, width]
    # input: mask[height, width]
    img_trans = image.copy().transpose(1, 2, 0)
    height = img_trans.shape[0]
    width = img_trans.shape[1]
    center = (int(width/2), int(height/2))

    angle = np.random.randint(0, 179)
    scale = 1.0

    trans = cv2.getRotationMatrix2D(center, angle , scale)
    image2 = cv2.warpAffine(img_trans, trans, (width, height))
    if mask is None: return image2.transpose(2, 0, 1)

    # print(mask.shape)
    mask2 = cv2.warpAffine(mask.copy(), trans, (width, height))
    return image2.transpose(2, 0, 1), mask2

def colour_jitter(image: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray,
                                                              Optional[np.ndarray]]:
    _, height, width = image.shape
    zitter = np.zeros_like(image)

    for channel in range(zitter.shape[0]):
        noise = np.random.randint(0, 30, (height, width))
        zitter[channel, :, :] = noise

    image = image + zitter
    if mask is None: return image
    return image, mask
