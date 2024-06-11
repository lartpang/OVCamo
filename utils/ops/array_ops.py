# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np


def minmax(data_array: np.ndarray, up_bound: float = None) -> np.ndarray:
    """
    ::

        data_array = (data_array / up_bound)
        if min_value != max_value:
            data_array = (data_array - min_value) / (max_value - min_value)

    :param data_array:
    :param up_bound: if is not None, data_array will devided by it before the minmax ops.
    :return:
    """
    if up_bound is not None:
        data_array = data_array / up_bound
    max_value = data_array.max()
    min_value = data_array.min()
    if max_value != min_value:
        data_array = (data_array - min_value) / (max_value - min_value)
    return data_array


def save_array_as_image(data_array: np.ndarray, save_name: str, save_dir: str, to_minmax: bool = False):
    """
    save the ndarray as a image

    Args:
        data_array: np.float32 the max value is less than or equal to 1
        save_name: with special suffix
        save_dir: the dirname of the image path
        to_minmax: minmax the array
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    if data_array.dtype != np.uint8:
        if data_array.max() > 1:
            raise Exception("the range of data_array has smoe errors")
        data_array = (data_array * 255).astype(np.uint8)
    if to_minmax:
        data_array = minmax(data_array, up_bound=255)
        data_array = (data_array * 255).astype(np.uint8)
    cv2.imwrite(save_path, data_array)


def resize(image_array: np.ndarray, height, width, interpolation=cv2.INTER_LINEAR):
    h, w = image_array.shape[:2]
    if h == height and w == width:
        return image_array

    resized_image_array = cv2.resize(image_array, dsize=(width, height), interpolation=interpolation)
    return resized_image_array


def ms_resize(img, scales, base_h=None, base_w=None, interpolation=cv2.INTER_LINEAR):
    assert isinstance(scales, (list, tuple))
    if base_h is None:
        base_h = img.shape[0]
    if base_w is None:
        base_w = img.shape[1]
    return [resize(img, height=int(base_h * s), width=int(base_w * s), interpolation=interpolation) for s in scales]
