from scipy import linalg
import numpy as np
import os, torch
from tqdm import tqdm


@torch.no_grad()
def _trace_sqrtm_product(C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
    # Tr sqrtm(C1 @ C2) = Tr sqrt( C1^{1/2} C2 C1^{1/2} )
    s, U = torch.linalg.eigh(C1)                 # C1 = U diag(s) U^T
    s = s.clamp_min(0)
    C1h = (U * s.sqrt()) @ U.t()                 # C1^{1/2}
    M   = C1h @ C2 @ C1h
    w   = torch.linalg.eigvalsh((M + M.t()) * 0.5).clamp_min(0)
    return w.sqrt().sum()

@torch.no_grad()
def calc_fid_stats(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    # 모두 float64 + 동일 device로 정렬
    C1 = torch.as_tensor(sigma1, dtype=torch.float64)
    device = C1.device
    C2 = torch.as_tensor(sigma2, dtype=torch.float64).to(device)
    m1 = torch.as_tensor(mu1,    dtype=torch.float64).to(device).flatten()
    m2 = torch.as_tensor(mu2,    dtype=torch.float64).to(device).flatten()

    D = m1.numel()
    I = torch.eye(D, dtype=torch.float64, device=device)

    # 대칭화 + 정칙화
    C1 = (C1 + C1.t()) * 0.5 + eps * I
    C2 = (C2 + C2.t()) * 0.5 + eps * I

    diff = m1 - m2
    tr_covmean = _trace_sqrtm_product(C1, C2)
    fid = diff.dot(diff) + torch.trace(C1) + torch.trace(C2) - 2.0 * tr_covmean
    return float(fid)

@torch.no_grad()
def calc_fid_pt_dir(pt_dir: str, mu, sigma, eps: float = 1e-6, num=100000, key="inception_feature") -> float:
    # pt_dir에서 'inception_feature'를 모아서 mu1, sigma1 추정 후 FID 계산
    X = []
    for f in tqdm(os.listdir(pt_dir)[:num]):
        if f.endswith(".pt"):
            v = torch.load(os.path.join(pt_dir, f), map_location="cpu").get(key)
            if v is not None:
                X.append(torch.as_tensor(v, dtype=torch.float64).flatten())
    if len(X) < 2:
        raise ValueError("need >=2 features")

    X   = torch.stack(X, 0)                 # [N, D]
    mu1 = X.mean(0)
    Xc  = X - mu1
    sigma1 = (Xc.t() @ Xc) / (X.shape[0] - 1)  # 불편추정

    return calc_fid_stats(mu1, sigma1, mu, sigma, eps=eps)
    #return calculate_frechet_distance(mu1, sigma1, mu, sigma, eps=eps)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)