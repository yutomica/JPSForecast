import numpy as np

def custom_asymmetric_mse(preds, train_data):
    """
    上振れ見逃し（y > y_pred）を重く罰する非対称MSE
    """
    y_true = train_data.get_label()
    residual = y_true - preds
    
    # ペナルティ倍率の設定
    # α: 実際の方が高い（上振れを見逃した）場合の罰の重さ
    # β: 予測の方が高い（過大評価した）場合の罰の重さ
    alpha = 3.0  
    beta = 1.0   
    
    # Gradient (1次微分: dL/d(preds))
    grad = np.where(residual > 0, 
                    -2.0 * alpha * residual,  # y > y_pred (上振れ見逃し)
                    -2.0 * beta * residual)   # y <= y_pred (過大評価)
    
    # Hessian (2次微分: d^2L/d(preds)^2)
    hess = np.where(residual > 0, 2.0 * alpha, 2.0 * beta)
    
    # サンプルウェイトの適用
    weight = train_data.get_weight()
    if weight is not None:
        grad *= weight
        hess *= weight
        
    return grad, hess

def custom_asymmetric_mse_eval(preds, train_data):
    """
    上振れ見逃し（y > y_pred）を重く罰する非対称MSE（評価・Early Stopping用）
    """
    y_true = train_data.get_label()
    residual = y_true - preds
    
    alpha = 3.0  
    beta = 1.0   
    
    # 損失の計算
    loss = np.where(residual > 0, alpha * (residual ** 2), beta * (residual ** 2))
    
    # サンプルウェイトの適用
    weight = train_data.get_weight()
    if weight is not None:
        loss *= weight
        return 'asym_mse', np.sum(loss) / np.sum(weight), False
        
    # 返り値: (メトリクス名, 評価値, is_higher_better)
    return 'asym_mse', np.mean(loss), False
