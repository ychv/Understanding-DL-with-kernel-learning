from src.get_data import inject_label_noise

class Trainer:
    """
    Class to train a model and help for plotting curves
    """

    def __init__(self,model) -> None:
        self.model=model
        self.train_sizes=[int(10**i) for i in range(2,5)]
        self.noise_ratios=[0.0,0.01,0.1]
        self.model_type=model.model_type
        self.epoch_nb=20
        pass

    def set_data(self,X_train,y_train,X_test,y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.model.set_train_data(X_train,y_train)
        self.X_test=X_test
        self.y_test=y_test
        pass


    def train_epochs(self,epoch_nb=None, checkpoints=None):

        """
        Train the model for different number of epochs.
        """

        if epoch_nb is None:
            epoch_nb=self.epoch_nb

        if self.model_type!='optimizer':
            raise ValueError("train_epochs is only available for 'optimizer' model type.")

        metrics_dict={'rkhs_norm':[],
                      'test_mse':[],
                      'train_mse':[],
                      'accuracy':[],
                      'classification_error':[],
                      'var':[]

        }
        
        if checkpoints is None:
            if epoch_nb is None: epoch_nb = 20
            checkpoints = range(5, epoch_nb + 1, 5)
        
        current_epoch = 0
        sorted_checkpoints = sorted(list(set(checkpoints)))

        for cp in sorted_checkpoints:
            steps = cp - current_epoch
            self.model.fit_step(step_epochs=steps)
            print(f"completed fit at epoch {cp}")
            current_epoch = cp
            self._log_metrics(metrics_dict)
            print(f"completed logging at epoch {cp}")
            metrics_dict['var'].append(current_epoch)

        return metrics_dict

    def train_size(self,train_sizes=None):
        """
        Train the model with different training sizes.
        """
        if train_sizes is None:
            train_sizes=self.train_sizes

        metrics_dict={'rkhs_norm':[],
                      'test_mse':[],
                      'train_mse':[],
                      'accuracy':[],
                      'classification_error':[],
                      'var': train_sizes

        }

        for train_sz in train_sizes:
            new_train_X=self.X_train[:train_sz]
            new_train_y=self.y_train[:train_sz]
            self.model.set_train_data(new_train_X,new_train_y)
            self.model.fit()
            self._log_metrics(metrics_dict)

        return metrics_dict
    
    def train_noise(self,noise_ratios=None):
        """
        Train the model with different noise ratios.
        """
        if noise_ratios is None:
            noise_ratios=self.noise_ratios

        metrics_dict={'rkhs_norm':[],
                      'test_mse':[],
                      'train_mse':[],
                      'accuracy':[],
                      'classification_error':[],
                      'var':noise_ratios
        }

        for noise_rt in noise_ratios:

            noisy_y=inject_label_noise(self.y_train,noise_rt)
            self.model.set_train_data(self.X_train,noisy_y)
            self.model.fit()
            self._log_metrics(metrics_dict)

        return metrics_dict
    
    def _log_metrics(self, metrics_dict):
        """
        Log various metrics during training.
        """

        y_pred = self.model.predict(self.X_test)
        
        y_true = self.y_test.to(y_pred.device)

        # Test metrics
        test_mse = self.model.compute_mse_loss(y_true, y_pred)
        accuracy = self.model.compute_accuracy(y_true, y_pred)
        classification_error = self.model.compute_classification_error(y_true, y_pred)

        # Train metrics
        subset_size = min(2000, self.model.X_train.size(0))
        X_train_sub = self.model.X_train[:subset_size].to(y_pred.device)
        y_train_sub = self.model.y_train[:subset_size].to(y_pred.device)
        
        y_pred_train = self.model.predict(X_train_sub)
        
        train_mse = self.model.compute_mse_loss(y_train_sub, y_pred_train)
        rkhs_norm = self.model.compute_rkhs_norm()

        metrics_dict['rkhs_norm'].append(rkhs_norm)
        metrics_dict['test_mse'].append(test_mse)
        metrics_dict['train_mse'].append(train_mse)
        metrics_dict['accuracy'].append(accuracy)
        metrics_dict['classification_error'].append(classification_error)
        
        print(f"Stats -> Test Acc: {accuracy:.2f} | Test MSE: {test_mse:.4f}")

    
