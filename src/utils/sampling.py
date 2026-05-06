import numpy as np
import pandas as pd

def apply_sampling(df, interval):
    if interval <= 1:
        return df
    print(f" [apply_sampling] Applying sampling (Date-interval): interval={interval} days")
    
    df = df.sort_values(['scode', 'date'])
    scodes = df['scode'].values
    dates = df['date'].values
    keep_mask = np.zeros(len(df), dtype=bool)
    
    last_scode = None
    last_date = np.datetime64('1900-01-01')
    interval_td = np.timedelta64(interval, 'D')
    
    for i in range(len(df)):
        if scodes[i] != last_scode or (dates[i] - last_date) >= interval_td:
            keep_mask[i] = True
            last_scode = scodes[i]
            last_date = dates[i]
            
    sampled_df = df[keep_mask].copy()
    return sampled_df


def apply_target_stratified_sampling(
    df: pd.DataFrame,
    target_col: str,
    date_col: str = 'date',
    scode_col: str = 'scode',
    mode: str = 'mode_1',
    center_keep_ratio: float = 0.25,
    other_keep_ratio: float = 1.0,
    weight_dict: dict = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    ターゲット変数の性質に応じた層化サンプリングまたは重み付けを実行します。
    Args:
        df (pd.DataFrame): 対象のデータフレーム。
        target_col (str): ターゲット変数名。
        date_col (str): 日付カラム名。
        scode_col (str): 銘柄コードカラム名。
        mode (str): 動作モード。
            - 'mode_1': Tailは100%保持、Centerをサンプリング、Otherは100%保持。
            - 'mode_2': Tailは100%保持、CenterとOtherをそれぞれ指定したレートでサンプリング。
            - 'mode_3': サンプリングせず、'sample_weight'列を付与する。
            - 'mode_ap_severe': AP_severeの閾値に基づき詳細な重み付けを行う。
        center_keep_ratio (float): Center部分の保持率 (mode_1, mode_2)。
        other_keep_ratio (float): Other部分の保持率 (mode_2)。
        weight_dict (dict): 各層に割り当てる重みの辞書 (mode_3)。例: {'tail': 2.0, 'center': 0.5, 'other': 1.0}
        random_state (int): 乱数シード。
    Returns:
        pd.DataFrame: 処理後のデータフレーム。mode_3, mode_ap_severe の場合は 'sample_weight' 列が追加される。
    """
    if target_col not in df.columns:
        print(f"  [Target-Stratified Sampling] Warning: target column '{target_col}' not found. Skipping.")
        if mode in ['mode_3', 'mode_ap_severe']:
            if 'sample_weight' not in df.columns:
                df['sample_weight'] = 1.0
        return df

    if mode == 'mode_ap_severe':
        # --- Mode: ap_severe_aligned ---
        # 定義に基づき、詳細な重み付けを行う。
        conditions = [
            (df[target_col] >= -0.02) & (df[target_col] <= 0.0), # center
            (df[target_col] > -0.05) & (df[target_col] < -0.02), # others
            (df[target_col] > -0.07) & (df[target_col] <= -0.05), # severe_5pct
            (df[target_col] > -0.10) & (df[target_col] <= -0.07), # severe_7pct
            (df[target_col] <= -0.10) # severe_10pct
        ]
        weights = [0.10, 1.00, 3.00, 4.00, 5.00]
        sw = np.select(conditions, weights, default=1.0)

        # 平均で正規化 (normalize_sample_weight_by_train_mean: true)
        if len(sw) > 0 and sw.mean() > 0:
            sw = sw / sw.mean()

        df['sample_weight'] = sw
        print(f"  [Target-Stratified Sampling] Mode: {mode}, Target: {target_col}")
        print(f"    - Applied normalized AP_severe-aligned weights.")
        return df

    if weight_dict is None:
        weight_dict = {'tail': 2.0, 'center': 0.5, 'other': 1.0}

    np.random.seed(random_state)
    original_len = len(df)
    
    is_tail = pd.Series(False, index=df.index)
    is_center = pd.Series(False, index=df.index)

    if target_col == 'target_tac_vol_scaled_asym_return_clipped':
        # テール：日次のクロスセクションにおいて、ターゲット値が上位20%に入るサンプル、または絶対値が +1.5 以上のサンプル
        def get_top_20(group):
            return group >= group.quantile(0.8)
        
        is_top_20 = df.groupby(date_col)[target_col].transform(get_top_20)
        is_abs_large = df[target_col].abs() >= 1.5
        is_tail = is_top_20 | is_abs_large
        
        # センター：ターゲット値が -1.5 〜 +0.5 の「退屈な値動き」のサンプル
        is_center = (df[target_col] >= -1.5) & (df[target_col] <= 0.5)

    elif target_col == 'target_str_sharpe_adj':
        # テール：日次のクロスセクションにおいて、ターゲットの絶対値（正負両方）が上位・下位あわせて30%に入るサンプル
        def get_abs_top_30(group):
            return group.abs() >= group.abs().quantile(0.7)
            
        is_tail = df.groupby(date_col)[target_col].transform(get_abs_top_30)
        
        # センター：ターゲット値がゼロ付近（Sharpeがほぼゼロ）のサンプル
        # （ここでは絶対値が0.5以下のものを「ゼロ付近」とみなしています）
        is_center = df[target_col].abs() < 0.5

    elif target_col == 'target_tac_max_neg_path':
        # テール：ターゲット値（最大下落率）が -0.05（-5%）より悪い（数値として小さい）サンプル
        is_tail = df[target_col] < -0.05
        
        # センター：ターゲット値が 0.0 〜 -0.02 に収まる、無風または微小な押し目程度のサンプル
        is_center = (df[target_col] >= -0.02) & (df[target_col] <= 0.0)

    elif target_col == 'target_str_mdd':
        # テール：ターゲット値（MDD）が 0.15（15%）以上のサンプル、または前週比でMDDが急拡大したサンプル
        temp_df = df[[scode_col, date_col, target_col]].copy()
        temp_df = temp_df.sort_values(by=[scode_col, date_col])
        # 前週（5営業日）比でのMDD拡大。0.05（5%）以上の拡大を「急拡大」とみなす
        temp_df['diff_5d'] = temp_df.groupby(scode_col)[target_col].diff(5)
        is_spike = temp_df['diff_5d'].reindex(df.index) >= 0.05
        
        is_tail = (df[target_col] >= 0.15) | is_spike
        
        # センター：ターゲット値（MDD）が 0.0 〜 0.05 未満の、60日間でほとんど下落しなかったサンプル
        is_center = (df[target_col] >= 0.0) & (df[target_col] < 0.05)

    else:
        print(f"  [Target-Stratified Sampling] Target '{target_col}' is not configured for sampling. Skipping.")
        if mode == 'mode_3':
            if 'sample_weight' not in df.columns:
                df['sample_weight'] = 1.0
        return df

    # テールに該当するものは必ず保持するため、センター判定から除外
    is_center = is_center & ~is_tail

    # Otherを定義 (TailでもCenterでもないもの)
    is_other = ~(is_tail | is_center)

    # ログ出力
    print(f"  [Target-Stratified Sampling] Mode: {mode}, Target: {target_col}")
    print(f"    - Original size    : {original_len:,}")
    print(f"    - Tail             : {is_tail.sum():,}")
    print(f"    - Center           : {is_center.sum():,}")
    print(f"    - Other            : {is_other.sum():,}")

    if mode == 'mode_3':
        # --- Mode 3: 重み付け ---
        weights = pd.Series(1.0, index=df.index)
        weights[is_tail] = weight_dict.get('tail', 2.0)
        weights[is_center] = weight_dict.get('center', 0.5)
        weights[is_other] = weight_dict.get('other', 1.0)

        df['sample_weight'] = weights

        print(f"    - Assigned weights: Tail={weight_dict.get('tail', 2.0)}, Center={weight_dict.get('center', 0.5)}, Other={weight_dict.get('other', 1.0)}")
        print(f"    - Sampled size     : {original_len:,} (Dropped 0.0% overall)")
        return df

    # --- Mode 1 & 2: サンプリング ---
    rand_vals = np.random.rand(len(df))

    # Centerのサンプリング
    center_drop_mask = is_center & (rand_vals >= center_keep_ratio)

    # Otherのサンプリング (mode_1では other_keep_ratio=1.0 なので常にFalse)
    other_drop_mask = is_other & (rand_vals >= other_keep_ratio)

    drop_mask = center_drop_mask | other_drop_mask
    keep_mask = ~drop_mask

    sampled_df = df[keep_mask].copy()
    sampled_len = len(sampled_df)
    drop_ratio = 1.0 - (sampled_len / original_len) if original_len > 0 else 0.0

    if mode == 'mode_2':
        print(f"    - Keep Ratios: Center={center_keep_ratio:.2f}, Other={other_keep_ratio:.2f}")
    else:  # mode_1
        print(f"    - Keep Ratios: Center={center_keep_ratio:.2f}, Other=1.00")

    print(f"    - Sampled size     : {sampled_len:,} (Dropped {drop_ratio:.1%} overall)")

    return sampled_df


def apply_2d_matrix_weight(df: pd.DataFrame, return_col: str, cost_buffer: float = 0.003) -> np.ndarray:
    """
    ターゲットリターンの値と日次クロスセクション順位の2次元マトリックスに基づき、サンプルウェイトを計算します。
    Args:
        df (pd.DataFrame): 'date' カラムと return_col カラムを含むデータフレーム。
        return_col (str): 5日リターン（価格比 P_{t+5}/P_t）のカラム名。
        cost_buffer (float): コストバッファの閾値。デフォルトは 0.003 (0.3%)。単なる売買コストではなく、「CSG上位をTrue Alphaとして強く学習させるための最低実現リターン閾値」
    Returns:
        np.ndarray: 計算されたサンプルウェイトの配列。
    """
    if return_col not in df.columns:
        print(f"  [2D-Matrix Weight] Warning: return column '{return_col}' not found. Returning uniform weights.")
        return np.ones(len(df))
    raw_return = np.log(df[return_col]) # raw_return: log(target_ret_5d)
    rank_pct = df.groupby('date')[return_col].transform(lambda x: x.rank(pct=True, ascending=True)) # 日次CS順位
    weights = np.ones(len(df))
    # true_alpha: rank_pct >= 0.80 and raw_return > cost_buffer
    mask_true_alpha = (rank_pct >= 0.80) & (raw_return > cost_buffer)
    weights[mask_true_alpha] = 1.5
    # moderate_winner: 0.60 <= rank_pct < 0.80 and raw_return > cost_buffer
    mask_mod_winner = (rank_pct >= 0.60) & (rank_pct < 0.80) & (raw_return > cost_buffer)
    weights[mask_mod_winner] = 1.2
    # rank_trap: rank_pct >= 0.80 and raw_return <= 0.0
    mask_rank_trap = (rank_pct >= 0.80) & (raw_return <= 0.0)
    weights[mask_rank_trap] = 0.8
    # center: 0.30 <= rank_pct < 0.60
    mask_center = (rank_pct >= 0.30) & (rank_pct < 0.60)
    weights[mask_center] = 0.9
    # clear_loser: rank_pct < 0.20 and raw_return < -cost_buffer
    mask_clear_loser = (rank_pct < 0.20) & (raw_return < -cost_buffer)
    weights[mask_clear_loser] = 1.0
    # other: 上記以外 (デフォルトの 1.0)
    print(f"  [2D-Matrix Weight] Applied weights based on {return_col} (cost_buffer={cost_buffer})")
    print(f"    - True Alpha: {mask_true_alpha.sum():,}, Moderate Winner: {mask_mod_winner.sum():,}")
    print(f"    - Rank Trap: {mask_rank_trap.sum():,}, Center: {mask_center.sum():,}")
    print(f"    - Clear Loser: {mask_clear_loser.sum():,}, Others: {(weights == 1.0).sum():,}")
    return weights