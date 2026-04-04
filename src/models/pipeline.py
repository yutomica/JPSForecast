import joblib
import numpy as np
import mlflow

class FoldPipeline:
    """1つのフォールドの『前処理 + モデル』を保持する最小単位"""
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

class EnsembleInferencePipeline(mlflow.pyfunc.PythonModel):
    """全フォールドのペアを管理し、アンサンブル予測を行う"""
    def __init__(self, fold_pipelines, col_indices):
        self.fold_pipelines = fold_pipelines # List[FoldPipeline]
        self.col_indices = col_indices

    def predict(self, context, model_input):
        """
        MLflow pyfunc interface.
        model_input: A pandas DataFrame with the features.
        """
        all_preds = []
        
        # 各フォールドの『ペア』ごとに個別に推論
        for fp in self.fold_pipelines:
            # 1. そのフォールド固有の統計量で前処理
            # Preprocessor is expected to handle a DataFrame for inference.
            X = fp.preprocessor.transform(model_input)
            # 2. そのフォールドのモデルで予測
            preds = fp.model.predict(X)
            all_preds.append(preds)
        
        # 全フォールドの予測値を平均（アンサンブル）
        return np.mean(all_preds, axis=0)

    def save(self, path):
        """
        モデルパイプラインを保存する。
        PyTorchモデルとsklearnモデルの混在に対応するため、torch.saveを使用する。
        LGBMなど、pickle化に問題があるモデルはjoblibで別途保存するなどの工夫が必要になる場合がある。
        """
        joblib.dump(self, path)
        print(f"Pipeline saved to {path} using joblib.dump")

    @classmethod
    def load(cls, path, device='cpu'):
        return joblib.load(path)