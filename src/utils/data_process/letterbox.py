import numpy as np
from typing import Tuple, Iterable
import cv2

def letterbox(img: np.ndarray, size: Tuple[int] = (224,224), fill_value: int = 128, return_extra_args = False):
    assert img.ndim == 2 or img.ndim == 3

    img = img.copy()
    img_dim = img.ndim

    if img.ndim == 2:
        img = img[...,np.newaxis]

    h, w, c = img.shape
    scale = min(size[0]/h, size[1]/w)
    n_h, n_w = int(h*scale), int(w*scale)

    img = cv2.resize(img, (n_w, n_h), interpolation= cv2.INTER_LANCZOS4)
    if img.ndim == 2:
        img = img[...,np.newaxis]

    pad = np.zeros([*size, c], dtype = img.dtype)
    pad.fill(fill_value)
    
    offset_h = (size[0] - n_h) //2
    offset_w = (size[1] - n_w) //2
    pad[offset_h:offset_h + n_h, offset_w:offset_w + n_w,:] = img

    if img_dim == 2:
        pad = pad[...,0]

    if return_extra_args:
        return pad, (offset_w, offset_h), scale
    return pad
