import torch


def sample_gumbel_diff(*shape):
    eps = 1e-20
    u1 = torch.rand(shape)
    u2 = torch.rand(shape)
    diff = torch.log(torch.log(u2+eps)/torch.log(u1+eps)+eps)
    return diff


def gumbel_sigmoid(logits, T=1.0, hard=False):
    g = sample_gumbel_diff(*logits.shape)
    y = (g + logits) / T
    s = torch.sigmoid(y)
    if hard:
        s_hard = s.round()
        s = (s_hard - s).detach() + s
    return s


def prob_to_logit(probs):
    return -torch.log((1-probs)/(probs+1e-8))


def left_child_idx(idx):
    return str(2*int(idx) + 1)


def right_child_idx(idx):
    return str(2*int(idx) + 2)


def parent_idx(idx):
    return str((int(idx)-1) // 2)


def return_perturbed(x, low, high):
    y = torch.rand_like(x) * (high - low) + low
    return x * y
