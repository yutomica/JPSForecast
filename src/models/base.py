from abc import ABC, abstractmethod

class BaseModelWrapper(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass