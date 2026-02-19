import pandas as pd
import numpy as np
import MySQLdb

TABLE_NAME = 'jps.sp_d'      
START_DATE = '2015-01-01' 

class DataLoader:
    """MySQLdbを使用してデータを抽出するクラス"""
    def __init__(self):
        self.conn = None
        self._connect()

    def _connect(self):
        try:
            self.conn = MySQLdb.connect(
                user='root',
                passwd='root',
                host='127.0.0.1',
                port=3306,
                charset='utf8',
            )
        except MySQLdb.Error as e:
            print(f"Error connecting to MySQL: {e}")
            raise

    def get_all_symbols(self):
        query = """
            WITH t1 AS (SELECT DISTINCT scode FROM jps.sp_d WHERE market != 'TOKYO PRO MARKET'),
            t2 AS (SELECT DISTINCT LEFT(Code,4) as scode,Sector33Code as sector33_code FROM org.listed_info)
            SELECT t1.scode,t2.sector33_code
            FROM t1 LEFT JOIN t2 ON t1.scode=t2.scode
        """
        try:
            df = pd.read_sql(query, self.conn)
            df = df.dropna(subset=['sector33_code'])
            df['sector33_code'] = df['sector33_code'].astype(str)
            return df
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return []

    def get_latest_symbols(self):
        query = f"SELECT DISTINCT scode FROM jps.scode_list WHERE scode not in ('0002') and market != 'その他' ORDER BY scode"
        try:
            df = pd.read_sql(query, self.conn)
            return df['scode'].tolist()
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return []
    
    def fetch_all_close_data(self):
        print("Fetching all close data for sector index creation (Optimized)...")
        query = f"""
            SELECT date, scode, close 
            FROM {TABLE_NAME} 
            WHERE mcode = 'T' AND date >= '{START_DATE}'
        """
        try:
            return pd.read_sql(query, self.conn, parse_dates=['date'])
        except Exception as e:
            print(f"Error fetching all close data: {e}")
            return pd.DataFrame()

    def fetch_batch_data(self, symbols, start_date=None):
        if not symbols:
            return pd.DataFrame()
        if start_date is None:
            start_date = START_DATE
        symbols_str = ",".join([f"'{s}'" for s in symbols])
        query = f"""
            SELECT date, scode, market, open, high, low, close, volume, volume_p 
            FROM {TABLE_NAME} 
            WHERE scode IN ({symbols_str}) 
            AND date >= '{start_date}' AND mcode = 'T'
            ORDER BY scode, date
        """
        try:
            return pd.read_sql(query, self.conn, parse_dates=['date'])
        except Exception as e:
            print(f"Error fetching batch data: {e}")
            return pd.DataFrame()

    def fetch_topix_data(self, start_date=None):
        if start_date is None:
            start_date = START_DATE
        query = f"""
            SELECT Date as date, Close as close 
            FROM org.indices_topix
            WHERE Date >= '{START_DATE}'
            ORDER BY date
        """
        try:
            df = pd.read_sql(query, self.conn, parse_dates=['date'])
            if not df.empty:
                df['Market_Return'] = np.log(df['close'] / df['close'].shift(1))
                df['Market_Trend_Idx'] = df['close'] / df['close'].rolling(25).mean()
                df['Market_HV_20'] = df['Market_Return'].rolling(20).std()
            return df
        except Exception as e:
            print(f"Error fetching TOPIX data: {e}")
            return pd.DataFrame()
    
    def fetch_sector_return(self, start_date=None):
        if start_date is None:
            start_date = START_DATE
        query = f"""
            SELECT * FROM jps.sector_return
            WHERE date >= '{start_date}'
        """
        try:
            df = pd.read_sql(query, self.conn, parse_dates=['date'])
            df['sector33_code'] = df['sector33_code'].astype(str)
            return df
        except Exception as e:
            print(f"Error fetching sector return data: {e}")
            return pd.DataFrame()
    
    def fetch_n225_data(self, start_date=None):
        if start_date is None:
            start_date = START_DATE
        query = f"""
            SELECT Date as date, Close as close 
            FROM jps.sp_d
            WHERE Date >= '{START_DATE}' AND scode = '0001' AND mcode = 'T'
            ORDER BY date
        """
        try:
            df = pd.read_sql(query, self.conn, parse_dates=['date'])
            if not df.empty:
                nikkei_ret = df['close'].pct_change()
                nikkei_hv = nikkei_ret.rolling(20).std()
                # HVの変化トレンド (今のHV - 20日前のHV)
                df['market_vol_change'] = nikkei_hv.diff()
            return df[['date', 'market_vol_change']]
        except Exception as e:
            print(f"Error fetching N225 data: {e}")
            return pd.DataFrame()

    # predict_daily.py 用の銘柄情報取得メソッドを追加
    def fetch_stock_info(self):
        """銘柄名、業種、最新終値を取得する"""
        query = "SELECT scode, sname, close, gyoshu FROM jps.scode_list"
        try:
            return pd.read_sql(query, self.conn)
        except Exception as e:
            print(f"Error fetching stock info: {e}")
            return pd.DataFrame()
    
    # 部門別取引情報系特徴量のローダー
    def fetch_investor_types(self, start_date=None):
        if start_date is None:
            start_date = START_DATE
        query = f"""
            SELECT PubDate, sum(FrgnBuy) as Foreign_Net_Buy, sum(IndBuy) as Individual_Net_Buy 
            FROM org.investor_types 
            WHERE PubDate >= '{START_DATE}'
            GROUP BY PubDate
        """
        try:
            df_trends = pd.read_sql(query, self.conn, parse_dates=['PubDate'])
            # 【修正箇所】Market_Return, Market_Trend_Idx, Market_HV_20 を計算して追加
            if not df_trends.empty:
                # マージ用のキーをセット
                # 株価データ(df_stock)の日付と突き合わせるのは「対象期間(Date)」ではなく「公開日(PublishedDate)」
                df_trends = df_trends.set_index('PubDate').sort_index()
                # 2. 必要なカラムのみ抽出
                # PublishedDateがIndexになっている
                df_trends = df_trends[['Foreign_Net_Buy', 'Individual_Net_Buy']]
                # 3. 日次カレンダーへのリサンプリングと前方埋め (ffill)
                # これにより、ある公開日から次の公開日まで、同じ値が継続する
                # 実際のバックテスト期間に合わせてreindexする
                # 例: 2018-01-01 から直近まで
                end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
                full_range = pd.date_range(start=start_date, end=end_date, freq='D')
                # 日次データに合わせてリインデックスし、値を前方埋めする
                # 例: 木曜(Pub) -> 金曜(穴埋め) -> 月曜(穴埋め) ... -> 次の木曜(新データ)
                df_daily = df_trends.reindex(full_range, method='ffill')
                # 期間設定: 短期トレンド(60日) と 長期水準(250日) の2つを採用
                windows = [60, 250]
                feature_cols = []
                for window in windows:
                    suffix = f"_{window}" # カラム名の接尾辞 (例: _60, _250)
                    # 海外投資家 Zスコア
                    mean_f = df_daily['Foreign_Net_Buy'].rolling(window).mean()
                    std_f = df_daily['Foreign_Net_Buy'].rolling(window).std()
                    col_name_f = f'Market_Foreign_Z{suffix}'
                    df_daily[col_name_f] = (df_daily['Foreign_Net_Buy'] - mean_f) / std_f
                    feature_cols.append(col_name_f)
                    # 個人投資家 Zスコア
                    mean_i = df_daily['Individual_Net_Buy'].rolling(window).mean()
                    std_i = df_daily['Individual_Net_Buy'].rolling(window).std()
                    col_name_i = f'Market_Individual_Z{suffix}'
                    df_daily[col_name_i] = (df_daily['Individual_Net_Buy'] - mean_i) / std_i
                    feature_cols.append(col_name_i)
                # Momentum（勢いの変化）は「短期(60)」のZスコアの変化を見るのが最も有効
                # 直近の資金流入加速を検知するため
                df_daily['Market_Foreign_Diff'] = df_daily['Market_Foreign_Z_60'].diff(5)
                feature_cols.append('Market_Foreign_Diff')
                # 海外投資家動向トレンド (4週移動平均)
                df_daily['overseas_flow_trend'] = df_daily['Foreign_Net_Buy'].rolling(20).mean()
                feature_cols.append('overseas_flow_trend')
                # フロー加速度
                flow = df_daily['overseas_flow_trend']
                df_daily['flow_accel'] = flow - flow.rolling(5).mean()
                feature_cols.append('flow_accel')
                # DataFrameを整えて返す
                df_daily.index.name = 'date'
                return df_daily[feature_cols].reset_index()
        except Exception as e:
            print(f"Error fetching investor types data: {e}")
            return pd.DataFrame()
        
    def fetch_financial(self, start_date=None):
        """
        財務データ(org.financials)を取得する
        戦略モデル構築仕様書に基づき、以下のカラムをマッピングして取得する:
        - published_date: DiscDate (開示日)
        - scode: Code
        - eps: FEPS (予想EPS / Forward PER用)
        - bps: BPS (実績BPS / PBR用)
        - dps: FDivAnn (予想年間配当 / 配当利回り用)
        - sales: Sales (実績売上高 / 成長率計算用)
        - operating_profit: OP (実績営業利益 / 利益率計算用)
        - net_income: NP (実績純利益 / ROA計算用)
        - total_assets: TA (実績総資産 / ROA, 自己資本比率用)
        - equity: Eq (実績自己資本 / ROE, 自己資本比率用)
        - shares_outstanding: ShOutFY (発行済株式数 / 時価総額計算用)
        """
        if start_date is None:
            start_date = START_DATE
        query = f"""
            SELECT 
                DiscDate as published_date,
                LEFT(Code,4) as scode,
                FEPS as eps,
                BPS as bps,
                FDivAnn as dps,
                Sales as sales,
                OP as operating_profit,
                FOP as operating_profit_forecast,
                NP as net_income,
                TA as total_assets,
                Eq as equity,
                ShOutFY as shares_outstanding,
                CFO as operating_cf
            FROM org.financials
            WHERE DiscDate >= '{start_date}'
            ORDER BY Code, DiscDate
        """
        try:
            # 日付パースを含めてDataFrameとして返す
            df = pd.read_sql(query, self.conn, parse_dates=['published_date'])
            # 会社予想データの欠損補完
            # 銘柄ごとにグループ化し、時間をさかのぼって欠損を埋める
            cols_forecast = ['eps', 'operating_profit_forecast', 'dps']
            df[cols_forecast] = df.groupby('scode')[cols_forecast].fillna(method='ffill')
            # 銘柄ごとに並べ替えた後、CFデータを前方埋め
            df['operating_cf'] = df.groupby('scode')['operating_cf'].fillna(method='ffill')
            # それでも欠損（上場直後など）の場合は 0 で埋める
            df['operating_cf'] = df['operating_cf'].fillna(0)
            # ffill
            df = df.sort_values(['scode','published_date']).set_index(['scode', 'published_date']).sort_index()
            df = df.groupby(level='scode').ffill().reset_index()
            df = df.drop_duplicates(subset=['scode', 'published_date'], keep='last')
            return df
        except Exception as e:
            print(f"Error fetching financial data: {e}")
            return pd.DataFrame()

    def fetch_margin_weekly(self, start_date=None):
        """
        信用取引週末残高データを取得する
        J-Quants API: /markets/trades/spec/margin/weekly 対応
        
        Returns:
            pd.DataFrame: [date, scode, long_margin_trade_balance_share, short_margin_trade_balance_share]
        """
        if start_date is None:
            start_date = START_DATE
        # ※ テーブル名はご自身のDB環境に合わせて修正してください (例: jps.margin_weekly)
        # J-Quantsのカラム名(PascalCase)を、特徴量エンジニアリング用の変数名(snake_case)に変換します
        query = f"""
            SELECT 
                Date as date,
                LEFT(Code, 4) as scode,
                LongVol as long_margin_trade_balance_share,
                ShrtVol as short_margin_trade_balance_share
            FROM org.margin_interest
            WHERE Date >= '{start_date}'
            ORDER BY Date, Code
        """
        try:
            df = pd.read_sql(query, self.conn, parse_dates=['date'])
            # 数値型への確実な変換（None対策）
            cols = ['long_margin_trade_balance_share', 'short_margin_trade_balance_share']
            df[cols] = df[cols].fillna(0).astype(float)
            return df
        except Exception as e:
            print(f"Error fetching Margin Weekly data: {e}")
            return pd.DataFrame()

    def fetch_short_selling_sector(self, start_date=None):
        """
        業種別空売り比率データを取得する
        J-Quants API: /markets/short_selling 対応
        
        Returns:
            pd.DataFrame: [date, sector33_code, selling_volume_ratio]
        """
        if start_date is None:
            start_date = START_DATE

        # ※ テーブル名はご自身のDB環境に合わせて修正してください (例: jps.short_selling_sector)
        query = f"""
            SELECT Date as date, 
                   S33 as sector33_code,
                   SellExShortVa,
                   ShrtWithResVa,
                   ShrtNoResVa
            FROM org.short_ratio
            WHERE Date >= '{start_date}'
            ORDER BY Date, S33
        """
        try:
            df = pd.read_sql(query, self.conn, parse_dates=['date'])
            df['SellExShortVa'] = df['SellExShortVa'].fillna(0).astype(float)
            df['ShrtWithResVa'] = df['ShrtWithResVa'].fillna(0).astype(float)
            df['ShrtNoResVa'] = df['ShrtNoResVa'].fillna(0).astype(float)
            df['selling_volume_ratio'] = [None if z is None or z == 0 else (x + y) / z for x, y, z in zip(df['ShrtWithResVa'], df['ShrtNoResVa'], df['SellExShortVa'])]
            # sector33_codeを文字列型に統一（結合時のキー不一致防止）
            df['sector33_code'] = df['sector33_code'].astype(str)
            return df[['date', 'sector33_code', 'selling_volume_ratio']]
        except Exception as e:
            print(f"Error fetching Short Selling data: {e}")
            return pd.DataFrame()


    def close(self):
        if self.conn:
            self.conn.close()