import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import datetime
import numpy as np
import talib 
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor 
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots 

# --- 1. Streamlit 頁面設定 ---
st.set_page_config(layout="wide", page_title="台灣個股智能分析系統 (V4.8)")

# --- 2. 數據獲取與緩存 (自動連網抓取) ---

@st.cache_data(ttl=24*3600) 
def fetch_history_data(stock_id, days=180):
    """自動從 yfinance 抓取歷史股價"""
    ticker = f"{stock_id}.TW"
    end_date = datetime.date.today()
    start_date = end_date - timedelta(days=days + 60) 
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        return None
    return data.tail(days) 

@st.cache_data(ttl=3*3600) 
def fetch_tse_chip_data(stock_id):
    """自動從 TWSE 抓取三大法人買賣超數據"""
    query_date = datetime.datetime.now().strftime("%Y%m%d")
    url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={query_date}&selectType=ALLBUT0999&response=json"
    
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        json_data = res.json()
        
        if json_data.get('stat') != 'OK':
            return {"error": f"TWSE查無資料 ({json_data.get('msg', '資料可能尚未更新或為假日')})"}
        
        df = pd.DataFrame(json_data['data'], columns=json_data['fields'])
        target_stock = df[df['證券代號'] == stock_id]
        
        if target_stock.empty:
            return {"error": f"❌ 找不到 {stock_id} 當日籌碼資料"}
            
        data = target_stock.iloc[0]
        def clean_volume(s):
            # 確保欄位是字串，並移除千分位逗號後轉為千張
            return int(str(data['三大法人買賣超股數']).replace(',', '')) / 1000 
        
        # 為了避免籌碼資料的欄位名稱太長，統一用字典回傳
        chip_data_result = {
            "日期": query_date,
            "股票名稱": data['證券名稱'],
            "三大法人合計 (千張)": int(str(data['三大法人買賣超股數']).replace(',', '')) / 1000,
            "外資買賣超 (千張)": int(str(data['外資自營商買賣超股數']).replace(',', '')) / 1000,
            "投信買賣超 (千張)": int(str(data['投信買賣超股數']).replace(',', '')) / 1000,
        }
        return chip_data_result

    except Exception as e:
        return {"error": f"💀 籌碼數據獲取錯誤: {e}"}

@st.cache_data(ttl=3*3600)
def fetch_fundamentals(stock_id):
    """自動從 yfinance 抓取基本面數據"""
    ticker = yf.Ticker(f"{stock_id}.TW")
    try:
        info = ticker.info
        stock_name = info.get('longName', f'股票代號 {stock_id}') 
        return {
            "P/E Ratio (本益比)": info.get('forwardPE'), 
            "EPS (每股盈餘)": info.get('trailingEps'),
            "股息殖利率 (%)": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else None,
            "市值 (B)": round(info.get('marketCap') / 1_000_000_000, 2) if info.get('marketCap') else None,
            "stock_name": stock_name
        }
    except Exception:
        return {"stock_name": f'股票代號 {stock_id}'}


# --- 3. 分析與模型訓練邏輯 (V4.8 穩定性核心) ---

def calculate_indicators(data):
    """計算所有技術指標 (使用 TA-Lib)，V4.4 加入 TA-Lib 異常捕獲"""
    
    # 第一次強制清除 NaN 值 (雖然 V4.7 在 main 已經清理過一次)
    data = data.dropna()
    
    # 數據完整性檢查
    if len(data) < 60:
        st.warning(f"⚠️ {st.session_state.get('current_stock', '該股票')} 歷史數據量不足 {len(data)} 筆 (至少需要約 60 筆)，技術指標無法計算。")
        return pd.DataFrame() 

    try:
        # 提取 numpy 陣列並檢查 Inf/NaN
        close_prices = data['Close'].values.astype(float)
        high_prices = data['High'].values.astype(float)
        low_prices = data['Low'].values.astype(float)
        
        # 最終數值校驗 - 確保陣列中沒有 Inf（無限大）或 NaN
        if np.isinf(close_prices).any() or np.isnan(close_prices).any():
             st.error("❌ 數據清洗失敗：股價數據中包含無限大 (Inf) 或 NaN 值，無法計算指標。")
             return pd.DataFrame()

    except ValueError:
        st.error("❌ 數據型態轉換錯誤：股價數據中可能包含非數值字串或無效值。")
        return pd.DataFrame()


    # --- V4.4 核心修正：加入 Try-Except 區塊來處理頑固的 TA-Lib 錯誤 ---
    try:
        # 均線
        data['MA_5'] = talib.SMA(close_prices, timeperiod=5)
        data['MA_20'] = talib.SMA(close_prices, timeperiod=20)
        data['MA_60'] = talib.SMA(close_prices, timeperiod=60)
        
        # KD 指標
        data['K'], data['D'] = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=9, slowk_period=3, slowd_period=3)

        # MACD 指標
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # 布林通道 (BBands)
        data['BB_Upper'], data['BB_Mid'], data['BB_Lower'] = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # RSI
        data['RSI'] = talib.RSI(close_prices, timeperiod=14)
        
    except Exception as e:
        # 捕獲所有 TA-Lib 拋出的異常 (包括 "wrong dimensions")
        st.error(f"💀 TA-Lib 計算指標時發生致命錯誤: {e}。這可能是由於環境或數據極端異常引起，已跳過分析。")
        return pd.DataFrame()
    
    # 再次清除因 TA-Lib 產生的 NaN 值
    return data.dropna()


def prepare_prediction_features(data, chip_data, fundamentals):
    """建立機器學習特徵 (V3.0)"""
    df = data.copy()
    
    # 標籤：次日漲跌幅 (百分比)
    df['Price_Change_Label'] = df['Close'].pct_change(periods=-1) * 100
    
    # 特徵工程
    df['Feature_Volume'] = df['Volume'] 
    df['Feature_K_minus_D'] = df['K'] - df['D'] 
    df['Feature_Close_MA20_Diff'] = (df['Close'] - df['MA_20']) / df['MA_20'] * 100 
    df['Feature_MACD_Hist'] = df['MACD_Hist'] 
    df['Feature_BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid'] * 100 
    df['Feature_RSI'] = df['RSI']
    
    feature_cols = [col for col in df.columns if col.startswith('Feature_')]
    df = df.dropna()
    
    # 檢查歷史數據是否足以訓練模型
    if df.shape[0] == 0:
        # 返回空的特徵集，會在 train_and_predict 中被捕獲
        return {
            "Feature_Columns": feature_cols,
            "Latest_Features_DF": pd.DataFrame(),
            "Historical_Data_DF": pd.DataFrame(),
        }

    # 最新一日的特徵 (用於實時預測)
    latest_features = df[feature_cols].iloc[-1].to_frame().T.reset_index(drop=True)
    
    # 補充籌碼/基本面特徵 (最新數據)
    latest_features['Feature_Chip_Total'] = chip_data.get('三大法人合計 (千張)', 0) if "error" not in chip_data else 0
    latest_features['Feature_PE'] = fundamentals.get('P/E Ratio (本益比)', 0)
    
    # 更新 Feature Columns 以納入籌碼和基本面
    feature_cols.extend(['Feature_Chip_Total', 'Feature_PE'])
    
    return {
        "Feature_Columns": feature_cols,
        "Latest_Features_DF": latest_features[feature_cols],
        "Historical_Data_DF": df.drop(columns=[c for c in df.columns if not (c.startswith('Feature_') or c == 'Price_Change_Label')]),
    }

# ⚠️ Streamlit Session State 存儲模型和 Scaler，避免重複訓練
if 'model_params' not in st.session_state:
    st.session_state['model_params'] = {}

@st.cache_data(show_spinner=False)
def train_and_predict(data_bundle, stock_id):
    """[V3 核心] 訓練 XGBoost 並進行次日漲跌預測"""
    historical_df = data_bundle['Historical_Data_DF']
    latest_features_df = data_bundle['Latest_Features_DF']
    feature_cols = data_bundle['Feature_Columns']
    
    # 檢查是否已訓練或數據是否足夠
    if historical_df.shape[0] < 50:
        return {"predicted_change_pct": None, "error": "⚠️ 歷史數據不足 50 筆，無法訓練機器學習模型。"}
    
    # 進行訓練或從 Session State 載入
    if stock_id in st.session_state['model_params']:
        model = st.session_state['model_params'][stock_id]['model']
        scaler = st.session_state['model_params'][stock_id]['scaler']
    else:
        # 進行訓練
        X = historical_df[feature_cols]
        Y = historical_df['Price_Change_Label']
        
        # 檢查 X, Y 是否有 NaN 或 Inf
        if X.isnull().values.any() or Y.isnull().values.any() or np.isinf(X.values).any():
             return {"predicted_change_pct": None, "error": "❌ 機器學習數據清洗失敗：特徵中含有 NaN 或 Inf 值，無法訓練模型。"}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, shuffle=False)
        
        model = XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.05, random_state=42)
        model.fit(X_train, Y_train)
        score = model.score(X_test, Y_test)
        
        # 儲存模型和 scaler
        st.session_state['model_params'][stock_id] = {'model': model, 'scaler': scaler, 'score': score}
        st.sidebar.success(f"✅ 模型訓練完成。測試集 R^2 分數: {score:.3f}")

    # 進行預測
    X_latest = latest_features_df[feature_cols]
    
    # 預測前再次檢查
    if X_latest.isnull().values.any() or np.isinf(X_latest.values).any():
        return {"predicted_change_pct": None, "error": "❌ 機器學習數據清洗失敗：最新特徵中含有 NaN 或 Inf 值，無法進行預測。"}

    X_latest_scaled = scaler.transform(X_latest)
    predicted_change_pct = model.predict(X_latest_scaled)[0]
    
    return {"predicted_change_pct": predicted_change_pct, "error": None}


# --- 4. 視覺化函式 ---

def plot_candlestick(data):
    """繪製 K 線圖與布林通道"""
    
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                        open=data['Open'],
                                        high=data['High'],
                                        low=data['Low'],
                                        close=data['Close'],
                                        name='K線'),
                        # 布林通道
                        go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='orange', width=1), name='上軌'),
                        go.Scatter(x=data.index, y=data['BB_Mid'], line=dict(color='gray', width=1), name='中軌'),
                        go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='orange', width=1), name='下軌')])
    
    fig.update_layout(title='股價 K 線圖與布林通道', xaxis_rangeslider_visible=False, height=500, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_macd_kd(data):
    """繪製 MACD 和 KD 指標圖"""
    
    # 創建子圖
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.5, 0.5])
    
    # MACD 圖
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], line=dict(color='blue'), name='MACD'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], line=dict(color='orange'), name='Signal'), row=1, col=1)
    # MACD 柱狀體
    bar_colors = np.where(data['MACD_Hist'] > 0, 'rgba(0,128,0,0.7)', 'rgba(255,0,0,0.7)')
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name='Hist', marker_color=bar_colors), row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=1, col=1)

    # KD 圖
    fig.add_trace(go.Scatter(x=data.index, y=data['K'], line=dict(color='red'), name='K'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['D'], line=dict(color='green'), name='D'), row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
    fig.update_yaxes(title_text="KD指標 (0-100)", range=[0, 100], row=2, col=1)
    
    fig.update_layout(title='MACD 與 KD 指標分析', height=500, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# --- 5. 輸出報告與建議 (V4.8 修正) ---

def generate_report(data, chip_data, fundamentals, buy_price, stop_loss_pct, take_profit_pct, prediction_result):
    """整合輸出所有分析結果"""
    latest = data.iloc[-1]
    stock_name = fundamentals.get('stock_name', f'股票代號 {st.session_state["current_stock"]}')
    
    st.header(f"💰 個股綜合分析報告 - {stock_name} ({st.session_state['current_stock']})")

    # A. 預測結果
    st.subheader("🔮 IV. 次日漲跌預測 (機器學習 V4.8)")
    
    pct = prediction_result['predicted_change_pct']
    if pct is not None:
        col_pct, col_status, col_price = st.columns(3)
        status = "📈 預期上漲" if pct > 0 else "📉 預期下跌"
        predicted_price = latest['Close'] * (1 + pct / 100)
        
        col_pct.metric("預期漲跌幅 (%)", f"{pct:.2f}%", delta=f"{pct:.2f}%", delta_color="inverse" if pct < 0 else "normal")
        col_status.metric("漲跌信號", status)
        col_price.metric("預測次日收盤價", f"TWD {predicted_price:.2f}")

    else:
        st.error(prediction_result['error'])


    st.subheader("📊 II. 技術面指標與圖表")
    
    # 檢查技術指標是否成功計算 
    indicators_available = 'BB_Lower' in data.columns
    
    if indicators_available:
        fig_candle = plot_candlestick(data)
        fig_macd_kd = plot_macd_kd(data)
        
        # V4.8 修正: 隔離 Plotly 繪製，避免 'removeChild' 錯誤
        col_kline, col_macd_kd = st.columns(2)
        
        with col_kline:
            st.plotly_chart(fig_candle, use_container_width=True)

        with col_macd_kd:
            st.plotly_chart(fig_macd_kd, use_container_width=True)
            
    else:
        st.warning("⚠️ 數據不足或 TA-Lib 錯誤，無法繪製完整的 K 線和指標圖。")


    # B. 停損停利建議
    st.subheader("🛡️ V. 股票停損停利建議")
    
    latest_close = latest['Close']
    
    pnl_pct = (latest_close - buy_price) / buy_price * 100
    
    st.metric("目前盈虧", f"{pnl_pct:.2f}%", delta=f"{pnl_pct:.2f}%", delta_color="inverse" if pnl_pct < 0 else "normal")
    
    advice = []
    
    # 停損/停利判斷 
    if pnl_pct >= take_profit_pct:
        advice.append(f"🟢 **獲利了結**：達成預設停利目標 ({take_profit_pct}%)")
    elif pnl_pct <= -stop_loss_pct:
        advice.append(f"🔴 **嚴守紀律**：跌破預設停損線 ({-stop_loss_pct}%)")

    # 技術面停損/停利判斷 (僅在指標可用時執行)
    if indicators_available:
        latest_bb_lower = latest['BB_Lower']
        latest_k = latest['K']
        latest_d = latest['D']
        latest_macd_hist = latest['MACD_Hist']
        
        if latest_close < latest_bb_lower:
             advice.append("⚠️ **技術警示**：股價跌破布林通道下軌，波動性增大。")
        if latest_k < latest_d and latest_k < 50:
            advice.append("✨ **潛在買點**：KD 低檔死亡交叉，若 K 值超跌，可關注。")
        if latest_macd_hist > 0 and latest_macd_hist < data['MACD_Hist'].iloc[-2]:
            advice.append("🚨 **動能減弱**：MACD 正柱體收斂，短期上漲動能減弱。")
        
    if advice:
        st.markdown("**綜合建議：**")
        for item in advice:
            st.markdown(f"* {item}")
    else:
        st.info("⭐ 股價仍在預期區間內，建議持續持有或觀察。")

    # C. 數據表格
    st.subheader("💡 I. 基本面與籌碼面數據")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 基本面 (價值評估)")
        # 移除 stock_name 欄位再顯示
        display_fundamentals = {k: v for k, v in fundamentals.items() if k != 'stock_name'}
        df_fundamentals = pd.DataFrame(display_fundamentals.items(), columns=["指標", "數值"])
        df_fundamentals = df_fundamentals.set_index("指標")
        st.dataframe(df_fundamentals, use_container_width=True)
        
    with col2:
        st.markdown("##### 籌碼面 (主力動向)")
        if "error" not in chip_data:
            df_chip = pd.DataFrame([chip_data])
            df_chip = df_chip[['日期', '三大法人合計 (千張)', '外資買賣超 (千張)', '投信買賣超 (千張)']]
            st.dataframe(df_chip.set_index('日期'), use_container_width=True)
            
            if chip_data.get('三大法人合計 (千張)', 0) > 0:
                 st.success("🟢 法人連續買超，籌碼相對集中。")
            else:
                 st.warning("🔴 法人賣超，須留意籌碼鬆動。")
        else:
            st.warning(chip_data['error'])


# --- 6. 介面主邏輯 ---

def main():
    st.title("📈 台灣個股智能分析系統 V4.8")
    st.markdown("---")
    st.sidebar.header("設置與查詢")

    with st.sidebar.form(key='analysis_form'):
        stock_ids_input = st.text_input("輸入股票代號 (多組請用逗號分隔)", value='2330, 2408')
        buy_price_input = st.number_input("輸入當初買入價格 (TWD)", min_value=1.0, value=580.0, format="%.2f")
        
        st.markdown("---")
        st.markdown("##### 風險管理設置")
        stop_loss_pct_input = st.number_input("停損百分比 (%)", min_value=1.0, value=5.0, format="%.1f")
        take_profit_pct_input = st.number_input("停利百分比 (%)", min_value=1.0, value=15.0, format="%.1f")
        
        submitted = st.form_submit_button("開始分析")

    if submitted:
        stock_list = [s.strip() for s in stock_ids_input.split(',') if s.strip()]
        
        if not stock_list:
            st.error("請至少輸入一個股票代號。")
            return

        for stock_id in stock_list:
            
            with st.spinner(f"正在分析 {stock_id}，請稍候... (連網抓取數據、訓練模型)"):
                
                # 1. 抓取數據 (自動連網)
                history_data = fetch_history_data(stock_id)
                chip_data = fetch_tse_chip_data(stock_id)
                fundamentals = fetch_fundamentals(stock_id)

                if history_data is None or history_data.empty:
                    st.error(f"❌ 無法獲取股票代號 {stock_id} 的歷史數據。請檢查代號或稍後再試。")
                    continue
                
                # 檢查 'Close' 欄位是否存在
                if 'Close' not in history_data.columns:
                    st.error(f"❌ 股票代號 {stock_id} 數據結構異常，缺少 'Close' 價格欄位。")
                    continue
                
                # 🌟 V4.7 核心修正: 強制將價格和成交量轉換為 float
                try:
                    history_data['Close'] = history_data['Close'].astype(float)
                    history_data['High'] = history_data['High'].astype(float)
                    history_data['Low'] = history_data['Low'].astype(float)
                    history_data['Volume'] = history_data['Volume'].astype(float)
                    
                    # 再次清除因轉換失敗產生的 NaN
                    history_data = history_data.dropna()
                    
                    # 強制檢查轉換後 DataFrame 是否為空
                    if history_data.empty:
                        st.error(f"❌ 股票代號 {stock_id} 數據在強制轉換為 float 後變為空集，無法進行分析。")
                        continue

                except ValueError as e:
                    st.error(f"❌ 股票代號 {stock_id} 數據強制轉換為 float 失敗，可能包含非數值字元。錯誤訊息: {e}")
                    continue


                # 2. 計算指標 (V4.4 數據檢查與清洗，包含異常捕獲)
                data_with_indicators = calculate_indicators(history_data.copy())
                
                if data_with_indicators.empty:
                    # 錯誤訊息已在 calculate_indicators 中顯示，這裡只需跳過
                    continue
                
                # 3. 準備特徵
                st.session_state['current_stock'] = stock_id
                
                prediction_data_bundle = prepare_prediction_features(data_with_indicators.copy(), chip_data, fundamentals)

                # 4. 訓練模型並預測
                prediction_result = train_and_predict(prediction_data_bundle, stock_id)

                # 5. 生成報告
                generate_report(
                    data_with_indicators, 
                    chip_data, 
                    fundamentals, 
                    buy_price_input, 
                    stop_loss_pct_input, 
                    take_profit_pct_input,
                    prediction_result
                )
                st.markdown("---") 
                st.markdown("---") 


if __name__ == "__main__":
    main()
