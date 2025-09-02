import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from keras.losses import MeanSquaredError
import plotly.graph_objects as go
import pickle
from rsier import RSI
from datetime import datetime, timedelta
import os
import gdown
import time

# Configuration
SYMBOL = "CW8.PA"
DRIVE_FILE_ID = "1eHvWadYri1NEZ2bRabsi6gyzyBAMeueO"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
LOCAL_CACHE = "cw8_cache.csv"
CACHE_MAX_AGE = 3600  # 1 heure en secondes
INPUT_FEATURES = ['day_of_week', 'week_of_year', 'month_of_year', 'quarter_of_year', 
                 'semester_of_year', 'lag_1_week', 'Vol_1_month', 'SMA20', 'SMA50', 
                 'RSI', 'return']

# Initialisation
st.set_page_config(page_title="CW8.PA Predictor", layout="wide")
st.title("Tracker CW8.PA - Price Prediction")
st.write("""
This application predicts future stock prices using a pre-trained LSTM model.

Data sources: [Yahoo Finance](https://finance.yahoo.com) | [GitHub Repo](https://github.com/aurvl/Stocks-prices-Prediction)
""")

# --- Data Loading System ---
def should_refresh_cache():
    """Check if cache needs refresh"""
    if not os.path.exists(LOCAL_CACHE):
        return True
    file_age = time.time() - os.path.getmtime(LOCAL_CACHE)
    return file_age > CACHE_MAX_AGE

def load_from_drive():
    """Load backup from Google Drive"""
    try:
        gdown.download(DRIVE_URL, LOCAL_CACHE, quiet=True)
        if os.path.exists(LOCAL_CACHE):
            return pd.read_csv(LOCAL_CACHE, index_col=0, parse_dates=True)
    except Exception as e:
        st.warning(f"Drive load failed: {str(e)}")
    return None

def get_fresh_data():
    """Get fresh data with rate limiting"""
    try:
        data = yf.download(SYMBOL, progress=False, timeout=10)
        if not data.empty:
            # Standardize column names
            data.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col 
                          for col in data.columns]
            data = data[data.index > '2018-04-17']
            data.to_csv(LOCAL_CACHE)
            return data
    except Exception as e:
        st.warning(f"Yahoo Finance error: {str(e)}")
    return None

@st.cache_data(ttl=CACHE_MAX_AGE)
def load_data():
    """Smart data loading with cache management"""
    if not should_refresh_cache():
        try:
            cached_data = pd.read_csv(LOCAL_CACHE, index_col=0, parse_dates=True)
            if not cached_data.empty:
                return cached_data
        except:
            pass
    
    fresh_data = get_fresh_data()
    if fresh_data is not None:
        return fresh_data
    
    return load_from_drive()

# --- Main Data Loading ---
data = load_data()
if data is None or data.empty:
    st.error("No data available from any source")
    st.stop()

# --- Feature Engineering ---
def prepare_features(data):
    """Generate all required features"""
    data = data.copy()
    st.write(f"These are the inputs : {data.columns}\n\n")
    data['dat'] = pd.to_datetime(data.index)
    data['day_of_week'] = data['dat'].dt.dayofweek
    data['week_of_year'] = data['dat'].dt.isocalendar().week
    data['month_of_year'] = data['dat'].dt.month
    data['quarter_of_year'] = data['dat'].dt.quarter
    data['semester_of_year'] = data['dat'].dt.quarter.apply(lambda x: 1 if x < 3 else 2)
    data = data.rename(columns={'Close_CW8.PA': 'price', 'Volume_CW8.PA': 'Volume'})
    st.write(f"These are the inputs : {data.columns}")
    data = data[['price', 'Volume', 'day_of_week', 'week_of_year', 'month_of_year', 
                'quarter_of_year', 'semester_of_year']].copy()
    return data

with st.spinner("Preparing data..."):
    data = prepare_features(data)
    st.success(f"Data loaded until {data.index[-1].strftime('%d/%m/%Y')}")

# --- User Interface ---
st.sidebar.header("Settings")
pred_days = st.sidebar.slider("Prediction days", 1, 20, 10)
refresh_btn = st.sidebar.button("Force Refresh Data")

if refresh_btn:
    st.cache_data.clear()
    os.remove(LOCAL_CACHE)
    st.experimental_rerun()

# --- Data Visualization ---
col1, col2 = st.columns([3, 1])
with col1:
    st.line_chart(data['price'], use_container_width=True)
with col2:
    st.metric("Last Price", f"{data['price'].iloc[-1]:.2f}€")
    st.metric("Volume", f"{data['Volume'].iloc[-1]:,}")

# --- Prediction System ---
def generate_future_dates(last_date, days):
    """Generate business days only"""
    dates = []
    i = 1
    while len(dates) < days:
        next_date = last_date + timedelta(days=i)
        if next_date.weekday() < 5:
            dates.append(next_date)
        i += 1
    return dates

def prepare_prediction_data(data, days):
    """Prepare combined historical + future data"""
    last_date = data.index[-1]
    future_dates = generate_future_dates(last_date, days)
    
    future_data = pd.DataFrame(index=future_dates)
    future_data['price'] = np.nan
    for col in ['day_of_week', 'week_of_year', 'month_of_year', 
                'quarter_of_year', 'semester_of_year']:
        future_data[col] = future_data.index.to_series().apply(
            lambda x: getattr(x, col) if hasattr(x, col) else x.month)
    
    combined = pd.concat([data, future_data])
    combined['lag_1_week'] = combined['price'].shift(5)
    combined['Vol_1_month'] = combined['Volume'].shift(20)
    combined['SMA20'] = combined['price'].rolling(20).mean()
    combined['SMA50'] = combined['price'].rolling(50).mean()
    combined['RSI'] = RSI(combined['price'])
    combined['return'] = combined['price'].pct_change()
    
    return combined.iloc[49:]

# --- Model Prediction ---
if st.button("Generate Predictions"):
    with st.spinner("Loading model..."):
        with open('src/preprocessor.pkl', 'rb') as f:
            scaler, target_scaler = pickle.load(f)
        
        model = load_model('src/CW8_pred_model.h5', 
                         custom_objects={'mse': MeanSquaredError()})
        model.compile(optimizer='adam', loss='mse')
    
    combined_data = prepare_prediction_data(data, pred_days)
    future_dates = combined_data[combined_data['price'].isna()].index
    
    with st.spinner("Making predictions..."):
        ddf = pd.DataFrame(scaler.transform(combined_data[INPUT_FEATURES].dropna()),
                          columns=INPUT_FEATURES, index=combined_data.dropna().index)
        
        predictions = []
        for date in future_dates:
            last_seq = ddf.loc[:date].iloc[-10:].values.reshape(1, 10, -1)
            pred = model.predict(last_seq)[0][0]
            pred_price = target_scaler.inverse_transform([[pred]])[0][0]
            combined_data.loc[date, 'price'] = pred_price
            predictions.append((date, pred_price))
            
            # Update features
            combined_data.loc[date, 'lag_1_week'] = combined_data['price'].shift(7).loc[date]
            combined_data.loc[date, 'Vol_1_month'] = combined_data['Volume'].shift(20).loc[date]
            combined_data.loc[date, 'SMA20'] = combined_data['price'].rolling(20).mean().loc[date]
            combined_data.loc[date, 'SMA50'] = combined_data['price'].rolling(50).mean().loc[date]
            combined_data['RSI'] = RSI(combined_data['price'])
            combined_data.loc[date, 'return'] = combined_data['price'].pct_change().loc[date]
            
            new_row = combined_data.loc[date, INPUT_FEATURES].values.reshape(1, -1)
            new_scaled = scaler.transform(new_row)
            ddf = pd.concat([ddf, pd.DataFrame(new_scaled, columns=INPUT_FEATURES, index=[date])])
    
    # Display results
    pred_df = pd.DataFrame(predictions, columns=["Date", "Predicted Price"])
    
    st.session_state["pred_df"] = pred_df
    
    st.success("Predictions completed!")
    st.dataframe(pred_df.set_index("Date"))

# Plot results
if "pred_df" in st.session_state:
    pred_df = st.session_state["pred_df"]

    # --- Sélecteur de période ---
    time_periods = {
        "All Data": data.index.min(),
        "YTD": pd.Timestamp(datetime(datetime.now().year, 1, 1)),
        "1 Year": datetime.now() - timedelta(days=365),
        "6 Months": datetime.now() - timedelta(days=180),
        "3 Months": datetime.now() - timedelta(days=90),
        "1 Month": datetime.now() - timedelta(days=30)
    }

    col1, col2, col3 = st.columns([4, 2, 1])
    with col2:
        selected_period = st.selectbox("Period", list(time_periods.keys()), label_visibility="collapsed")

    period_mask = (data.index >= time_periods[selected_period])

    # --- Détection de tendance ---
    start_pred = pred_df["Predicted Price"].iloc[0]
    end_pred = pred_df["Predicted Price"].iloc[-1]
    trend_color = "green" if end_pred >= start_pred else "red"

    # --- Dates combinées pour x-axis complète ---
    combined_dates = list(data.index[period_mask]) + list(pred_df["Date"])
    combined_prices = list(data['price'][period_mask]) + list(pred_df["Predicted Price"])

    # --- Tracé ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[period_mask], y=data['price'][period_mask],
                             name="Historical", line=dict(color='blue', width=1), mode='lines'))
    fig.add_trace(go.Scatter(x=[data.index[period_mask][-1]] + list(pred_df["Date"]),
                             y=[data['price'][period_mask][-1]] + list(pred_df["Predicted Price"]),
                             name="Predicted", line=dict(color=trend_color, width=1), mode='lines'))
    fig.add_trace(go.Scatter(x=combined_dates, y=combined_prices, mode='none', showlegend=False))

    fig.update_layout(
        title=f"Price Prediction - {selected_period} View",
        xaxis_title="Date",
        yaxis_title="Price (€)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Bouton de téléchargement ---
    csv = pred_df.to_csv().encode('utf-8')
    st.download_button("Download Predictions", csv, 
                       f"cw8_predictions_{datetime.now().date()}.csv", "text/csv")
    

st.markdown(":red[**NB: This app is for educational purposes only. Use at your own risk.**]")
st.write("Contact me at : *[Aurel VEHI](https://github.com/aurvl)*")
