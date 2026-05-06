import numpy as np

def custom_asymmetric_mse(asym_alpha=3.0, asym_beta=1.0, **kwargs):
    """
    上振れ見逃し（y > y_pred）を重く罰する非対称MSE（ファクトリ関数）
    """
    def _objective(preds, train_data):
        y_true = train_data.get_label()
        residual = y_true - preds
        
        # ペナルティ倍率の設定
        # asym_alpha: 実際の方が高い（上振れを見逃した）場合の罰の重さ
        # asym_beta: 予測の方が高い（過大評価した）場合の罰の重さ
        
        # Gradient (1次微分: dL/d(preds))
        grad = np.where(residual > 0, 
                        -2.0 * asym_alpha * residual,  # y > y_pred (上振れ見逃し)
                        -2.0 * asym_beta * residual)   # y <= y_pred (過大評価)
        
        # Hessian (2次微分: d^2L/d(preds)^2)
        hess = np.where(residual > 0, 2.0 * asym_alpha, 2.0 * asym_beta)
        
        # サンプルウェイトの適用
        weight = train_data.get_weight()
        if weight is not None:
            grad *= weight
            hess *= weight
            
        return grad, hess
    return _objective

def custom_asymmetric_mse_eval(asym_alpha=3.0, asym_beta=1.0, **kwargs):
    """
    上振れ見逃し（y > y_pred）を重く罰する非対称MSE（評価・Early Stopping用ファクトリ関数）
    汎用的な NumPy インターフェース (y_true, y_pred, dates=None) を返します。
    """
    def _eval(y_true, y_pred, dates=None):
        residual = y_true - y_pred
        
        # 損失の計算
        loss = np.where(residual > 0, asym_alpha * (residual ** 2), asym_beta * (residual ** 2))
        
        # LightGBMのような(name, value, is_higher_better)形式ではなく、スコアのみを返す
        return np.mean(loss)
        
    return _eval
