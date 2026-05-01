# Calculate L and dL/dy (dy/dW in /layer.py)
import numpy as np

class Loss:
    """Base class (meant to be ABC)"""
    
    def compute(self, y_true, y_pred):
        """
        Calculate total error (scalar) betweem target and prediction
        
        Args:
            y_true (np.ndarray): True label
                                Shape: (batch_size, output_dim).
            y_pred (np.ndarray): Prediction label. 
                                Shape: (batch_size, output_dim).
            
        Returns:
            float: average loss in a batch.
        """
        raise NotImplementedError("Subclasses must implement compute()")
    
    def gradient(self, y_true, y_pred):
        """
        Gradient calculation for Loss to y_pred (dL/dy_pred).
        
        Args:
            y_true (np.ndarray): True label
                                Shape: (batch_size, output_dim).
            y_pred (np.ndarray): Prediction label. 
                                Shape: (batch_size, output_dim).z
                                
        Returns:
            np.ndarray: Gradient with the same shape as y_pred (batch_size, output_dim).
        """
        raise NotImplementedError("Subclasses must implement gradient()")
    
class MSE(Loss):
    def compute(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    def gradient(self, y_true, y_pred):
        n = y_true.shape[0]
        return -2 * (y_true - y_pred) / n
    
class BCE(Loss):
    def compute(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def gradient(self, y_true, y_pred):
        n = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true - y_pred)/(y_pred * (1 - y_pred) * n)
    
class CCE(Loss):
    def compute(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def gradient(self, y_true, y_pred):
        n = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true/y_pred) / n
