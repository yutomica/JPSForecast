import joblib
import numpy as np

class FoldPipeline:
    """1つのフォールドの『前処理 + モデル』を保持する最小単位"""
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

class EnsembleInferencePipeline:
    """全フォールドのペアを管理し、アンサンブル予測を行う"""
    def __init__(self, fold_pipelines, col_indices):
        self.fold_pipelines = fold_pipelines # List[FoldPipeline]
        self.col_indices = col_indices

    def predict(self, data, row_indices):
        all_preds = []
        
        # 各フォールドの『ペア』ごとに個別に推論
        for fp in self.fold_pipelines:
            # 1. そのフォールド固有の統計量で前処理
            X = fp.preprocessor.transform(
                data, 
                row_indices=row_indices, 
                col_indices=self.col_indices
            )
            # 2. そのフォールドのモデルで予測
            preds = fp.model.predict(X)
            all_preds.append(preds)
        
        # 全フォールドの予測値を平均（アンサンブル）
        return np.mean(all_preds, axis=0)

    def save(self, path):
        """
        joblib ではなく、PyTorch の serialization を活用して保存する
        (PyTorch モデル以外の sklearn オブジェクトも混在可能)
        """
        import torch
        # 自作クラスや sklearn オブジェクトが含まれていても、
        # torch.save は pickle の上位互換として機能するため保存可能です
        torch.save(self, path)
        print(f"Pipeline saved to {path} using torch.save")

    @classmethod
    def load(cls, path, device='cpu'):
        import torch
        model = torch.load(path, map_location=device)
        return model