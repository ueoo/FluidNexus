from typing import NamedTuple

import numpy as np


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    near: float
    far: float
    time_idx: int
    timestamp: float
    pose: np.array
    hp_directions: np.array
    cxr: float
    cyr: float
    is_fake_view: bool = False
    real_image: np.array = None
