import mlflow.pyfunc
import numpy as np

class EnsembleModel(mlflow.pyfunc.PythonModel):
    def __init__(self, models):
        # 学習済みのモデルリスト（models）を保持
        self.models = models

    def predict(self, context, model_input):
        # 各モデルの予測値を取得して平均を取る（アンサンブル）
        preds = [model.predict(model_input) for model in self.models]
        return np.mean(preds, axis=0)