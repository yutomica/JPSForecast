from abc import ABC, abstractmethod

class BaseModelWrapper(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, model_idx=0, epoch_callback=None, train_dates=None, valid_dates=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass