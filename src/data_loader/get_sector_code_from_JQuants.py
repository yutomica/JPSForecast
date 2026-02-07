import requests
import json
import pandas as pd
import sys
import os

def get_sector_master_from_api(mail_address, password):
    """
    J-Quants APIにログインし、最新の銘柄一覧を取得して
    scodeと33業種コード(Sector33Code)の対応表を返す関数。
    
    Args:
        mail_address (str): J-Quants登録メールアドレス
        password (str): J-Quants登録パスワード
        
    Returns:
        pd.DataFrame: columns=['scode', 'sector33_code', 'sector33_name']
    """
    
    # --- 1. リフレッシュトークンの取得 ---
    auth_user_url = "https://api.jquants.com/v1/token/auth_user"
    headers = {"content-type": "application/json"}
    data = {
        "mailaddress": mail_address,
        "password": password
    }
    
    try:
        res_auth = requests.post(auth_user_url, headers=headers, data=json.dumps(data))
        res_auth.raise_for_status() # エラーなら例外発生
        refresh_token = res_auth.json().get('refreshToken')
        print("Success: Retrieved Refresh Token.")
    except Exception as e:
        print(f"Error: Failed to get Refresh Token. Check your ID/Password. \n{e}")
        return pd.DataFrame()

    # --- 2. IDトークンの取得 ---
    # Refresh Tokenを使って実際にAPIを叩くためのID Tokenを取得します
    auth_refresh_url = f"https://api.jquants.com/v1/token/auth_refresh?refreshtoken={refresh_token}"
    
    try:
        res_refresh = requests.post(auth_refresh_url)
        res_refresh.raise_for_status()
        id_token = res_refresh.json().get('idToken')
        print("Success: Retrieved ID Token.")
    except Exception as e:
        print(f"Error: Failed to get ID Token. \n{e}")
        return pd.DataFrame()

    # --- 3. 銘柄一覧(上場銘柄属性)の取得 ---
    # Freeプランで利用可能なエンドポイントです
    info_url = "https://api.jquants.com/v1/listed/info"
    headers = {"Authorization": f"Bearer {id_token}"}
    
    try:
        print("Fetching listed info from J-Quants API...")
        res_info = requests.get(info_url, headers=headers)
        res_info.raise_for_status()
        info_data = res_info.json().get('info')
    except Exception as e:
        print(f"Error: Failed to fetch listed info. \n{e}")
        return pd.DataFrame()

    # --- 4. データ整形 ---
    if not info_data:
        print("Error: No data returned from API.")
        return pd.DataFrame()

    df = pd.DataFrame(info_data)
    
    # 必要なカラムが存在するか確認
    required_cols = ['Code', 'Sector33Code', 'Sector33CodeName']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' missing in API response.")
            return pd.DataFrame()

    # データ抽出とリネーム
    # J-QuantsのCodeは5桁(例: 72030)の場合があるため、先頭4桁をscodeとして使用
    df['scode'] = df['Code'].str[:4]
    df['sector33_code'] = df['Sector33Code']
    df['sector33_name'] = df['Sector33CodeName']

    # 必要な列のみ保持
    df_sector = df[['scode', 'sector33_code', 'sector33_name']].copy()

    # 重複排除
    # 優先株や新株などで同じ4桁コードが複数行ある場合、最初の1つを採用（通常、本株が最初に来るため）
    df_sector = df_sector.drop_duplicates(subset=['scode'], keep='first')
    
    # scodeでソート
    df_sector = df_sector.sort_values('scode').reset_index(drop=True)

    print(f"Success: Sector master created. Total {len(df_sector)} symbols.")
    return df_sector

# --- 実行ブロック (テスト用) ---
if __name__ == "__main__":
    # 実際の運用時は環境変数や引数から取得することを推奨します
    # ここでは入力プロンプトで確認できるようにしています
    print("--- J-Quants API Sector Master Builder ---")
    my_mail = os.environ.get('JQ_MAIL')
    my_pass = os.environ.get('JQ_PASS')

    if my_mail and my_pass:
        df_master = get_sector_master_from_api(my_mail, my_pass)
        
        if not df_master.empty:
            print("\n--- Preview (Top 5) ---")
            print(df_master.head())
            
            # CSV保存（必要に応じて）
            df_master.to_csv("sector_master.csv", index=False)
    else:
        print("Skipped.")