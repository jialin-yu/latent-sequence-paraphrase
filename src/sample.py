# hard and soft gumbel-softmax sampling
# Implementation borrowed from https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def straight_through_softmax(logits):
    y = F.softmax(logits, dim=-1)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return y_hard - y.detach() + y

def straight_through_logits(logits):
    y = logits
    shape = logits.size()
    mask = F.one_hot(torch.argmax(logits, dim=-1), shape[-1])
    y_hard = mask*logits
    return y_hard - y.detach() + y

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

# def gumbel_topk_(logits, temperature, topk):
#     """
#     input: [*, n_class]
#     return: [*, n_class] an one-hot vector
#     """
#     vals, idx = torch.topk(logits, topk, dim=-1, sorted=False)
#     logits_ = torch.zeros_like(logits).scatter(-1, idx, vals)
#     msk = (logits_ == 0).float()
#     logits_[msk == 1] = -float("inf")
#     y_ = F.gumbel_softmax(logits_, temperature, False)
#     shape = y_.size()
#     _, ind = y_.max(dim=-1)
#     y_hard = torch.zeros_like(y_).view(-1, shape[-1]).scatter(1, ind.view(-1, 1), 1)
#     # y_hard.scatter_(1, ind.view(-1, 1), 1)
#     y_hard = y_hard.view(*shape)
#     return y_hard - (y_).detach() + y_

def gumbel_softmax_topk_(logits, temperature, topk, hard=True):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    vals, idx = torch.topk(logits, topk, dim=-1, sorted=False)
    logits_ = torch.zeros_like(logits).scatter(-1, idx, vals).float()
    msk = (logits_ == 0).float()
    logits_[msk == 1] = -float("inf")
    y_ = F.gumbel_softmax(logits_, temperature, False)

    shape = y_.size()
    _, ind = y_.max(dim=-1)
    y_hard = torch.zeros_like(y_).view(-1, shape[-1]).scatter(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)

    if hard:
        return y_hard - (y_).detach() + y_
    else:
        return y_


def gumbel_softmax_topk(logits, temperature, topk, hard=True):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    samples = []
    for i in range(10):
        y_s = F.gumbel_softmax(logits, temperature, False)
        samples.append(y_s.unsqueeze(-1))
    y = torch.mean(torch.cat(samples, dim=-1), dim=-1)
    # y = F.gumbel_softmax(logits, temperature, False)
    vals, idx = torch.topk(y, topk, dim=-1, sorted=False)
    y_ = torch.zeros_like(y).scatter(-1, idx, vals).float()
    msk = (y_ == 0).float()
    y_[msk == 1] = -float("inf")
    y__ = F.softmax(y_, dim=-1)
    shape = y__.size()
    _, ind = y__.max(dim=-1)
    y_hard = torch.zeros_like(y__).view(-1, shape[-1]).scatter(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)

    if hard:
        return y_hard - (y__).detach() + y__
    else:
        return y__

def softmax_topk(logits, topk):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    vals, idx = torch.topk(logits, topk, dim=-1, sorted=False)
    logits_ = torch.zeros_like(logits).scatter(-1, idx, vals).float()
    msk = (logits_ == 0).float()
    logits_[msk == 1] = -float("inf")
    y = F.softmax(logits_, dim=-1)
    return y

# def gumbel_softmax_topk_(logits, temperature, topk, num_samples):
#     """
#     input: [*, n_class]
#     return: [*, n_class] an one-hot vector
#     """
#     samples = []
#     for i in range(num_samples):
#         y_s = F.gumbel_softmax(logits, temperature, False)
#         samples.append(y_s.unsqueeze(-1))
#     y = torch.mean(torch.cat(samples, dim=-1), dim=-1)
#     vals, idx = torch.topk(y, topk, dim=-1, sorted=False)
#     y_ = torch.zeros_like(y).scatter(-1, idx, vals)
#     msk = (y_ == 0).float()
#     y_[msk == 1] = -float("inf")
#     y_ = F.softmax(y_, dim=-1)
    
#     return y_

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y