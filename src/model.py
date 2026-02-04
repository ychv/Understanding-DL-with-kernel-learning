import torch
from eigenpro2.models import KernelModel
from src.metrics import accuracy,mse,classification_error

class KernelRegressor:

    def __init__(self,kernel_func,sigma=1.0,model_type='interpolator'):

        self.kernel_func=kernel_func
        self.sigma=sigma
        self.mse_loss= lambda x,y: torch.cdist(x,y,p=2)
        self.model_type=model_type
        self.model=None

    def set_train_data(self,X_train,y_train):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X_train=X_train.float().to(device)
        self.y_train=y_train.float().to(device)
        self.K = None
        self.alpha = None
        self.eigen_V = None
        print(f"Training data set with {X_train.size(0)} samples and {X_train.size(1)} features on {device}.")


    def fit(self):
        if self.model_type=='interpolator':
            self.invers_fit()
        elif self.model_type=='optimizer':
            self.optim_fit()

        else:
            raise ValueError("Unknown model type, either 'interpolator' or 'optimizer' expected.")

        pass

    def fit_step(self,step_epochs):
        """
        Fit the model for a given number of epochs without reinitializing parameters.
        Only for otpimizer model type
        """
        if self.model_type!='optimizer':
            raise ValueError("fit_step is only available for 'optimizer' model type.")
        
        self.optim_fit(epochs=step_epochs)

        pass

    def invers_fit(self):
        """
        Solve for f* using the closed form solution of kernel regression.
        This method require to inverse the kernel matrix and might be computationally expensive.
        In the paper, it is stated that direct methods always provide an highly accurate interpolation.
        """
        if not hasattr(self,'X_train') or not hasattr(self,'y_train'):
            raise ValueError("Training data not set. Please use set_train_data method first.")
        
        device = self.X_train.device
        # One hot encode labels
        if self.y_train.dim() == 1:
            y_onehot = torch.nn.functional.one_hot(self.y_train.long(), 10).float().to(device)
        else:
            y_onehot = self.y_train.float().to(device)

        # Compute Kernel matrix used to find f*
        self.K = self.kernel_func(self.X_train, self.X_train, self.sigma)

        # Add small regularization for numerical stability
        self.K += 1e-8*torch.eye(self.K.size(0), device=device)

        # Inverse to find weights
        self.alpha=torch.linalg.solve(self.K,y_onehot)

    def optim_fit(self, epochs=10):
        """
        Solve for f* using EigenPro gradient descent optimization.
        """
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise ValueError("Training data not set.")

        n_classes = 10

        # Labels encoding
        if self.y_train.dim() == 1:

            y_onehot = torch.nn.functional.one_hot(self.y_train.long(), n_classes).float()
        else:
            y_onehot = self.y_train.float()

        device = self.X_train.device
        X = self.X_train.float().to(device)
        y = y_onehot.to(device)

    
        if self.model is None:
            print("Initialisation du modèle EigenPro...")
            kernel_fn = lambda x, z: self.kernel_func(x, z, sigma=self.sigma)
            self.model = KernelModel(kernel_fn, X, n_classes, device=device)

        # Fit the model with EigenPro
        self.model.fit(X, y, epochs=epochs,bs=256, mem_gb=32)
        print("EigenPro optimization completed.")

    def predict(self,X_test):
        """
        Use the calculated weights to predict on new data
        """
        device = self.X_train.device
        X_input = X_test.to(device).float()
        if self.model_type == 'interpolator':
            if self.alpha is None:
                raise ValueError("Model not fitted yet")
            
            K_test=self.kernel_func(X_input,self.X_train,self.sigma)
            y_pred=K_test@self.alpha
            
            return y_pred
        
        elif self.model_type == 'optimizer':
                if self.model is None:
                    raise ValueError("Model not fitted yet (EigenPro model is None).")
                

                return self.model.forward(X_input)
            
        else:
             raise ValueError("Unknown model type.")

    def compute_rkhs_norm(self):
        """
        Compute the RKHS norm of the learned function f*
        
        """
        if self.alpha is None:
            if self.model_type == 'optimizer':
                # RKHS norm computation for EigenPro is not implemented in this scope
                return 0.0
            raise ValueError("Model not fitted yet")
        
        # Use the fomula of the paper to compute the RHKS norm
        interaction_matrix=self.alpha.T @ self.K @ self.alpha

        return torch.sqrt(torch.trace(interaction_matrix))
    
    def compute_accuracy(self,y_true,y_pred):
        return accuracy(y_pred,y_true)
    
    def compute_mse_loss(self,y_true,y_pred):
        return mse(y_true,y_pred)
    
    def compute_classification_error(self,y_true,y_pred):
        return classification_error(y_true,y_pred)
    


    
    




    