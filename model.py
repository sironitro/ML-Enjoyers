from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    
    def __init__(self, name: str):
        self.name = name
        self.trained = False
        
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def export(self):
        pass
    
    @classmethod
    @abstractmethod
    def load():
        pass
    
    @abstractmethod
    def predict(sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        pass
    
    