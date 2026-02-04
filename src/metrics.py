import torch

def accuracy(y_pred,y_true):
    """
    Computes the accuracy between the prediction and true labels.
    """
    try :
        predicted_classes = torch.argmax(y_pred, dim=1)
        correct_predictions = (predicted_classes.view(-1) == y_true.view(-1))
    except ValueError:
        predicted_classes = torch.argmax(y_true, dim=1)
        correct_predictions = (predicted_classes.view(-1) == y_pred.view(-1))

    
    return correct_predictions.float().mean().item()

def precision(y_true,y_pred):
    """
    Computes the precision between the prediction and true labels.
    """

    correct=(y_true==y_pred).sum().item()
    total=(y_pred==1).sum().item()
    return correct/total

def mse(y_true,y_pred):
    if y_true.dim() == 1:
        y_true = torch.nn.functional.one_hot(y_true.long(), num_classes=y_pred.size(1)).float()
    
    return torch.nn.functional.mse_loss(y_pred, y_true).item()

def classification_error(y_true,y_pred):
    if y_pred.dim() > 1:
        y_pred = torch.argmax(y_pred, dim=1)
    incorrect=(y_true!=y_pred).sum().item()
    total=y_true.size(0)
    return incorrect/total
    
