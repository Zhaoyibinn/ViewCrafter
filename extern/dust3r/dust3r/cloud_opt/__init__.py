# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# global alignment optimization wrapper function
# --------------------------------------------------------
from enum import Enum

from .optimizer import PointCloudOptimizer
from .pair_viewer import PairViewer


class GlobalAlignerMode(Enum):
    PointCloudOptimizer = "PointCloudOptimizer"
    PairViewer = "PairViewer"


def global_aligner(dust3r_output, device, mode=GlobalAlignerMode.PointCloudOptimizer,dtu_path = None,**optim_kw):
    # extract all inputs
    view1, view2, pred1, pred2 = [dust3r_output[k] for k in 'view1 view2 pred1 pred2'.split()]
    # build the optimizer
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        net = PointCloudOptimizer(view1, view2, pred1, pred2, dtu_path = dtu_path,**optim_kw).to(device)
    elif mode == GlobalAlignerMode.PairViewer:
        net = PairViewer(view1, view2, pred1, pred2, **optim_kw).to(device)
    else:
        raise NotImplementedError(f'Unknown mode {mode}')

    return net
