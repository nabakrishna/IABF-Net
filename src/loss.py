import torch
import torch.nn as nn

class IABFTaskLoss(nn.Module):
    def __init__(self, weight_classification=1.0, weight_regression=2.0):
        super().__init__()
        self.weight_classification = weight_classification
        self.weight_regression = weight_regression
        self.classification_loss_fn = nn.BCEWithLogitsLoss()
        self.regression_loss_fn = nn.L1Loss()

    def forward(self, predictions, targets):
        predicted_classification = predictions[:, 0:1, :, :]
        target_classification = targets[:, 0:1, :, :]
        
        predicted_regression = predictions[:, 1:, :, :]
        target_regression = targets[:, 1:, :, :]
        
        classification_loss = self.classification_loss_fn(
            predicted_classification, 
            target_classification
        )
        regression_loss = self.regression_loss_fn(
            predicted_regression, 
            target_regression
        )
        
        total_loss = (self.weight_classification * classification_loss) + \
                     (self.weight_regression * regression_loss)
                     
        return total_loss, classification_loss, regression_loss