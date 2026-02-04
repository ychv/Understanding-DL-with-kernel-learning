import torch

def gaussian_kernel(x,y,sigma=20.0):
    """
    Computes the Gaussian kernel between two inputs.
    sigma: Spread parameter
    """
    dist = torch.cdist(x, y, p=2)**2
    return torch.exp(-dist/(2 * sigma**2))


def laplacian_kernel(x,y,sigma=20.0):
    """
    Computes the Laplacian kernel between two inputs.
    
    sigma: Spread parameter"""

    diff=torch.cdist(x,y,p=2)
    return torch.exp(-diff/sigma)
