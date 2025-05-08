from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    """
    An abstract class for our models in the ensemble.
    Subclasses must implement the core methods for training, exporting,
    loading, and making predictions.
    """
    def __init__(self, name: str):
        """
        Initializes the model with a name and sets the trained flag to False.

        Parameters:
        name (str): A unique identifier for the model instance.
        """
        self.name = name
        self.trained = False
        
    
    @abstractmethod
    def train_model(self):
        """
        Trains the model on available data.
        """
        pass
    
    @abstractmethod
    def export(self):
        """
        Exports the model's trained state/learned parameters to a file.
        """
        pass
    
    @classmethod
    @abstractmethod
    def load():
        """
        Loads a previously exported model from disk.

        Returns:
        Model: An instance of a subclass of Model with loaded parameters.
        """
        pass
    
    @abstractmethod
    def predict(sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        """
        Makes predictions for the given arrays of scientist and paper IDs.

        Parameters:
        sids (np.ndarray): An array of scientist identifiers.
        pids (np.ndarray): An array of paper identifiers.

        Returns:
        np.ndarray: An array of predicted values.
        """
        pass
    
    