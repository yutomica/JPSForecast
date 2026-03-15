import yaml
import pandas as pd
import re
from collections import defaultdict

def generate_feature_lists():
    tac_features = pd.read_csv('./scripts/feature_select_rough/candidates_tac.csv')['feature'].tolist()
    str_features = pd.read_csv('./scripts/feature_select_rough/candidates_str.csv')['feature'].tolist()

    # サフィックス定義
    SUFFIXES = ['CSZ', 'CSR', 'SNZ', 'TSZ_20D', 'TSR_252D', 'TSZ_60D', 'TSZ_120D', 'RAW']
    
    def parse_feature(f_name):
        # サフィックスを特定してベース名と分離
        for s in sorted(SUFFIXES, key=len, reverse=True):
            if f_name.endswith('_' + s):
                base = f_name[:-(len(s)+1)]
                return base, s
        return f_name, 'OTHER'

    def is_price_related(base_name):
        # 価格やリターンに直接由来する指標（RAW採用禁止対象）
        price_keywords = ['Return', 'Close', 'Price', 'PRC', 'RET', 'High', 'Low', 'Open']
        return any(k in base_name for k in price_keywords)

    def select_features(feature_list, target_type, model_group):
        families = defaultdict(list)
        for f in feature_list:
            base, suffix = parse_feature(f)
            families[base].append(suffix)
        selected = []
        # モデル別・ターゲット別の優先順位ロジック
        for base, available_suffixes in families.items():
            pool = []
            if model_group == 'LGBM':
                cap = 3
                if target_type == 'tac':
                    order = ['CSZ', 'CSR', 'TSZ_20D', 'SNZ']
                else: # str
                    order = ['SNZ', 'CSZ', 'TSR_252D', 'CSR']
            elif model_group == 'TabNet':
                cap = 2
                if target_type == 'tac':
                    order = ['CSZ', 'TSZ_20D', 'CSR']
                else:
                    order = ['SNZ', 'TSR_252D', 'CSZ']
            elif model_group == 'DeepTabular':
                cap = 2
                order = ['CSZ', 'SNZ', 'TSZ_20D'] if target_type == 'tac' else ['SNZ', 'CSZ', 'TSR_252D']
            elif model_group == 'TimeSeries': 
                cap = 2
                if target_type == 'tac':
                    order = ['TSZ_20D']
                else:
                    order = ['TSZ_60D', 'TSZ_120D', 'TSR_252D']
                # 非価格RAWの追加（ボラティリティやフローのみ）
                if 'RAW' in available_suffixes and not is_price_related(base):
                    pool.append(f"{base}_RAW")
            elif model_group == 'PureTS': 
                cap = 1
                if target_type == 'tac':
                    order = ['TSZ_20D']
                else:
                    order = ['TSR_252D']
                # 非価格RAWの追加（ボラティリティやフローのみ）
                if 'RAW' in available_suffixes and not is_price_related(base):
                    pool.append(f"{base}_RAW")
            # 優先順位に従ってピックアップ
            for s in order:
                if s in available_suffixes:
                    pool.append(f"{base}_{s}")
            
            selected.extend(pool[:cap])
            
        return sorted(list(set(selected)))

    # 各モデル・ターゲットのリスト作成
    configs = {
        'tac_LGBM': select_features(tac_features, 'tac', 'LGBM'),
        'tac_TabNet': select_features(tac_features, 'tac', 'TabNet'),
        'tac_DeepTabular': select_features(tac_features, 'tac', 'DeepTabular'),
        'tac_TimeSeries': select_features(tac_features, 'tac', 'TimeSeries'),
        'tac_PureTS': select_features(tac_features, 'tac', 'PureTS'),
        'str_LGBM': select_features(str_features, 'str', 'LGBM'),
        'str_TabNet': select_features(str_features, 'str', 'TabNet'),
        'str_DeepTabular': select_features(str_features, 'str', 'DeepTabular'),
        'str_TimeSeries': select_features(str_features, 'str', 'TimeSeries'),
        'str_PureTS': select_features(str_features, 'str', 'PureTS')
    }

    # カテゴリ変数のベース名リスト（実際のカテゴリ変数名に適宜変更してください）
    cat_features = [
        'EVT_IsMissingEPS_RAW',
        'GOV_market_segment_RAW',
        'GOV_sector33_code_RAW',
        'SEA_DayOfMonth_RAW',
        'SEA_DayOfWeek_RAW',
        'SEA_IsGotobi_RAW',
        'SEA_IsMonthEnd_RAW',
        'SEA_Quarter_RAW',
        'SEA_IsQuarterEnd_RAW',
    ]

    # YAMLファイルへの書き出し
    for name, flist in configs.items():
        # base名がcat_featuresに含まれるものをcat_colsとして抽出
        cat_cols = [f for f in flist if parse_feature(f)[0] in cat_features]
        
        output_dict = {
            "name": name,
            "feature_cols": flist,
            "cat_cols": cat_cols
        }

        with open(f'features_{name}_rough.yaml', 'w') as yf:
            yaml.dump(output_dict, yf, default_flow_style=False, sort_keys=False)
        print(f"Generated {name}.yaml: {len(flist)} features")

if __name__ == "__main__":
    generate_feature_lists()