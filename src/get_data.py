import torch
from torchvision import datasets, transforms

def get_mnist(n_samples=5000, flatten=True):
    """
    Loads MNIST, subsamples it and flattens images to vectors.
    """
    # Load full MNIST
    dataset=datasets.MNIST('./data', train=True, download=True,transform=transforms.ToTensor())
    
    # Subsample data
    indices=torch.randperm(len(dataset))[:n_samples]
    data=dataset.data[indices].float()/255.0
    targets=dataset.targets[indices]
    
    if flatten:
        data=data.view(data.size(0),-1)
        
    return data,targets

def inject_label_noise(targets,noise_ratio,n_classes=10):
    """
    Flips 'noise_ratio' fraction of labels to random classes in order to simulate label noise.
    """
    y_noisy=targets.clone()
    n_noise=int(len(targets)*noise_ratio)
    
    # Select random indices to corrupt
    noise_indices=torch.randperm(len(targets))[:n_noise]
    
    # Assign random labels
    random_labels=torch.randint(0,n_classes,(n_noise,))
    y_noisy[noise_indices]=random_labels
    
    return y_noisy



def get_fashion_mnist(n_samples=5000, flatten=True):
    """
    Loads Fashion-MNIST, subsamples it and flattens images to vectors.
    Output dim: (n_samples, 784) if flatten=True.
    """
    # Load full Fashion-MNIST
    dataset = datasets.FashionMNIST('./data', train=True, download=True)
    
    # Subsample data
    indices = torch.randperm(len(dataset))[:n_samples]
    # Note: dataset.data is already a Torch Tensor (N, 28, 28) for FashionMNIST
    data = dataset.data[indices].float() / 255.0 
    targets = dataset.targets[indices]
    
    if flatten:
        # (N, 28, 28) -> (N, 784)
        data = data.view(data.size(0), -1)
    else:
        # Add channel dimension: (N, 28, 28) to (N, 1, 28, 28)
        data = data.unsqueeze(1)
        
    return data, targets