import math
from typing import Union
import torch

from tensordict.tensordict import TensorDict
from torch import Tensor

from rl4co.utils.pylogger import get_pylogger
from rl4co.data.transforms import get_augment_function, StateAugmentation

log = get_pylogger(__name__)

def pdp_augmentation(xy: Tensor, reduce: bool = True, *args, **kw) -> Tensor:
    """
    Args:
        xy: [batch, graph, 2] tensor of x and y coordinates
    """
    # [batch, graph, 2]
    xy = xy[: xy.shape[0] // 2, ...] if reduce else xy
    num_loc = xy.shape[1]
    depot, pickup, delivery = xy[:, :1, :], xy[:, 1 : num_loc // 2 + 1, :], xy[:, num_loc // 2 + 1:, :]
    # augmnetations [batch, graph, 2]
    z0 = torch.cat((depot, pickup, delivery), dim=1)
    z1 = torch.cat((depot, delivery, pickup), dim=1)
    # [batch*2, graph, 2]
    aug_xy = torch.cat((z0, z1), dim=0)
    return aug_xy

def atsp_augmentation(dmat: Tensor, reduce: bool = True, *args, **kw) -> Tensor:
    """
    Args:
        dmat: [batch, graph, graph]
    """
    # [batch*2, graph, 2]
    dmat = dmat[: dmat.shape[0] // 2, ...] if reduce else dmat
    aug_dmat = torch.cat((dmat, dmat.mT), dim=0)
    return aug_dmat

class CompositeAugmentation(object):
    def __init__(
        self,
        num_augment_list: list,
        augment_fn_list: list, 
        first_aug_identity: bool = True,
        normalize: bool = False,
        feats: list = None,
    ):
        self.augment_list = []
        self.n_aug = 1
        for num_augment, augment_fn in zip(num_augment_list, augment_fn_list, strict=True):
            if augment_fn == "atsp":
                augment_fn = atsp_augmentation
                assert num_augment == 2
            elif augment_fn == "pdp":
                augment_fn = pdp_augmentation
                assert num_augment == 2

            augment = StateAugmentation(
                num_augment=num_augment,
                augment_fn=augment_fn,
                first_aug_identity=first_aug_identity,
                normalize=normalize,
                feats=feats,
            )

            self.augment_list.append(augment)
            self.n_aug *= max(num_augment, 1)

    def __call__(self, td: TensorDict) -> TensorDict:
        td_aug = td
        for augment in self.augment_list:
            td_aug = augment(td_aug)
        return td_aug

        