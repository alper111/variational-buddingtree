import torch


def gumbel_sigmoid(logit, tau=1.0, hard=False):
    u = torch.rand(logit.shape).to(logit.device)
    if torch.isinf(torch.log(u)).any():
        print("YES!")
    if torch.isinf(torch.log(1-u)).any():
        print("YES 2!")
    g = torch.log(u) - torch.log(1-u)
    noisy_logit = (g+logit)/tau
    y = torch.sigmoid(noisy_logit)
    if hard:
        y_hard = y.round()
        y = (y_hard - y).detach() + y
    return y


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
