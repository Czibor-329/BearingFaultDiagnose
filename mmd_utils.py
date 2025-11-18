import torch
from typing import List

def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigmas: List[float]) -> torch.Tensor:
    xx = (x ** 2).sum(dim=1, keepdim=True)
    yy = (y ** 2).sum(dim=1, keepdim=True).T
    dist2 = xx + yy - 2 * x @ y.T
    K = 0
    for s in sigmas:
        gamma = 1.0 / (2.0 * s * s)
        K = K + torch.exp(-gamma * dist2)
    return K

def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas: List[float]) -> torch.Tensor:
    Kxx = gaussian_kernel(x, x, sigmas)
    Kyy = gaussian_kernel(y, y, sigmas)
    Kxy = gaussian_kernel(x, y, sigmas)

    n = x.size(0)
    m = y.size(0)

    if n > 1:
        Kxx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
    else:
        Kxx = Kxx.sum() / (n * n + 1e-8)

    if m > 1:
        Kyy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
    else:
        Kyy = Kyy.sum() / (m * m + 1e-8)

    Kxy = Kxy.mean()
    mmd2 = Kxx + Kyy - 2 * Kxy
    return torch.clamp(mmd2, min=0.0)

def mk_sigmas_median(x: torch.Tensor, y: torch.Tensor, num_scales: int = 5):
    Z = torch.cat([x, y], dim=0)
    n = Z.size(0)
    idx = torch.randperm(n)[:min(n, 512)]
    Zs = Z[idx]
    d2 = (Zs**2).sum(1, keepdim=True) + (Zs**2).sum(1, keepdim=True).T - 2 * (Zs @ Zs.T)
    d = torch.sqrt(torch.clamp(d2, min=0) + 1e-12)
    med = torch.median(d).item()
    base = max(med, 1e-6)
    return [base * (2.0 ** k) for k in range(-(num_scales//2), (num_scales//2)+1)]