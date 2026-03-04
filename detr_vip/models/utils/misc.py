from typing import Optional, List

import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

class NestedTensor2D(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedImage # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor2D(cast_tensor, cast_mask)

    @property
    def device(self):
        return self.tensors.device

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    @property
    def shape(self):
        return {
            'tensors.shape': self.tensors.shape,
            'mask.shape': self.mask.shape
        }


def nested_2Dtensor_from_tensor_list(tensor_list: List[Tensor], padding_value=-1):
    # TODO make this more general
    requires_grad = tensor_list[0].requires_grad
    if tensor_list[0].ndim == 2:
        max_len = tensor_list[0].shape[0]
        for tensor in tensor_list:
            max_len = max_len if max_len > tensor.shape[0] else tensor.shape[0]
        batch_shape = [len(tensor_list), max_len, tensor_list[0].shape[1]]
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        # tensor = torch.ones(batch_shape, dtype=dtype, device=device) * padding_value
        # tensor.requires_grad = requires_grad

        mask = torch.ones((len(tensor_list), max_len), dtype=torch.bool, device=device)
        # mask.requires_grad = requires_grad
        new_tensor = []
        for t, m in zip(tensor_list, mask):
            new_pad_t = torch.cat((t, padding_value * torch.ones(max_len - t.shape[0], *t.shape[1:]).to(dtype).to(device)), 0)
            # new_pad_t[: t.shape[0]] = t
            new_tensor.append(new_pad_t)
            # pad_t[: t.shape[0]].copy_(t)
            m[: t.shape[0]] = False
        tensor = torch.stack(new_tensor)
    elif tensor_list[0].ndim == 1:
        max_len = tensor_list[0].shape[0]
        for tensor in tensor_list:
            max_len = max_len if max_len > tensor.shape[0] else tensor.shape[0]
        batch_shape = [len(tensor_list), max_len]
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.ones(batch_shape, dtype=dtype, device=device) * padding_value
        mask = torch.ones((len(tensor_list), max_len), dtype=torch.bool, device=device)
        for t, pad_t, m in zip(tensor_list, tensor, mask):
            pad_t[: t.shape[0]] = t
            # pad_t[: t.shape[0]].copy_(t)
            m[: t.shape[0]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor2D(tensor, mask)

def gather_logits(logits, detach=False):
    world_size = dist.get_world_size()
    local_n = torch.tensor([logits.shape[0]], dtype=torch.int64, device=logits.device)
    local_rank = dist.get_rank()

    # 收集所有进程的样本数
    all_n = torch.zeros(world_size, dtype=torch.int64, device=logits.device)
    dist.all_gather_into_tensor(all_n, local_n)

    # pad每个embeddings
    max_n = all_n.max().item()
    padded_logits = torch.zeros((max_n, *list(logits.shape[1:])), dtype=logits.dtype, device=logits.device)
    unpadded_flags = torch.zeros((max_n) , dtype=logits.dtype, device=logits.device).bool()

    padded_logits[:local_n] = logits.clone()
    unpadded_flags[:local_n] = True

    # 计算总样本数
    total_n = max_n * world_size

    # 预分配收集张量
    gathered_logits = torch.zeros((total_n, *list(logits.shape[1:])), dtype=logits.dtype, device=logits.device)
    gathered_flag = torch.zeros((total_n), dtype=logits.dtype, device=logits.device).bool()
    if detach:
        dist.all_gather_into_tensor(gathered_logits, padded_logits.detach())
        dist.all_gather_into_tensor(gathered_flag, unpadded_flags.detach())

        gathered_logits[local_rank * max_n:local_rank * max_n+local_n] = logits
    else:
        gathered_logits = torch.cat(all_gather(padded_logits))
        gathered_flag = torch.cat(all_gather(unpadded_flags))

    logits = gathered_logits[gathered_flag]
    return logits