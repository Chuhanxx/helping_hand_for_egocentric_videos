import torch

def clip_gradients(model, clip_grad=3):
    """from https://github.com/facebookresearch/dino/blob/main/main_dino.py"""
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip_grad / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms