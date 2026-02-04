import torch
import matplotlib.pyplot as plt

from src.kernels import gaussian_kernel, laplacian_kernel

def toy_visualization_side_by_side():

    N = 40
    sigma = 0.3   # Width of the kernel
    reg = 1e-8    # Tiny regularization
    outlier_idx = 20
    
    # Common Data Generation
    X = torch.linspace(-3, 3, N).view(-1, 1)
    y_true = torch.sin(X * 2) # Clean function
    X_test = torch.linspace(-3, 3, 500).view(-1, 1)

    # Initialize Plot (1 Row, 2 Columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scenarios: False = Clean, True = Corrupted
    scenarios = [False, True]
    titles = ["Scenario A: Clean Data", "Scenario B: Corrupted Data (1 Outlier)"]

    for i, is_corrupted in enumerate(scenarios):
        ax = axes[i]
        
        # 1. Prepare Target Data
        y_train = y_true.clone()
        if is_corrupted:
            y_train[outlier_idx] = -1  # Inject massive outlier
            
        y_target = y_train.view(-1, 1)

        # 2. Fit Models (Direct Math)
        # -- Gaussian Fit --
        K_g = gaussian_kernel(X, X, sigma) + reg * torch.eye(N)
        alpha_g = torch.linalg.solve(K_g, y_target)
        y_pred_g = gaussian_kernel(X_test, X, sigma) @ alpha_g

        # -- Laplacian Fit --
        K_l = laplacian_kernel(X, X, sigma) + reg * torch.eye(N)
        alpha_l = torch.linalg.solve(K_l, y_target)
        y_pred_l = laplacian_kernel(X_test, X, sigma) @ alpha_l

        # 3. Plotting on the specific axis (ax)
        # Plot True Function
        ax.plot(X, y_true, 'k--', alpha=0.4, label='True Function')
        
        # Plot Data Points
        ax.scatter(X, y_train, c='black', s=30, label='Data')
        
        # Highlight Outlier if present
        if is_corrupted:
            ax.scatter(X[outlier_idx], y_train[outlier_idx], 
                       c='red', s=150, marker='x', linewidth=3, label='Outlier')

        # Plot Model Predictions
        ax.plot(X_test, y_pred_g, 'b-', linewidth=1, label='Gaussian (Smooth)')
        ax.plot(X_test, y_pred_l, 'r-', linewidth=1, label='Laplacian (Spiky)')
        
        # Styling
        ax.set_title(titles[i], fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3.5, 3.5) # Fix y-limits to make comparison easier

    plt.suptitle(f"Inductive Bias Comparison (sigma={sigma})", fontsize=16)
    plt.tight_layout()
    plt.savefig('inductive_bias_comparison.png') # Saves the figure for your report
    plt.show()

if __name__ == "__main__":
    toy_visualization_side_by_side()