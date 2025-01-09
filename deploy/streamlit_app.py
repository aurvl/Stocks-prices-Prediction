import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
import plotly.graph_objects as go
import pickle

with open('src/preprocessor.pkl', 'rb') as f:
    preprocessors = pickle.load(f)

scaler, target_scaler = preprocessors

# Streamlit app title
st.title("WPEA Stock Price Prediction")
st.write("This application predicts the future stock prices for WPEA.PA using a pre-trained LSTM model.")

# Step 0: User Input
st.sidebar.header("Prediction Settings")
num_days = st.sidebar.slider("Number of days to predict:", min_value=1, max_value=30, value=30)
st.sidebar.write("Predicting for:", num_days, "days")

# Step 1: Download historical data
st.write("### Downloading Historical Data")
data = yf.download("WPEA.PA")
data.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in data.columns]
data = data[['Close_WPEA.PA', 'Volume_WPEA.PA']]
data['dat'] = data.index

# Step 2: Add feature columns to historical data
st.write("### Adding Feature Columns")
data['week_of_year'] = data['dat'].dt.isocalendar().week
data['month_of_year'] = data['dat'].dt.month
data['quarter_of_year'] = data['dat'].dt.quarter
data['lag_1_week'] = data['Close_WPEA.PA'].shift(7)
data['Vol_1_month'] = data['Volume_WPEA.PA'].shift(30)
data['MA10'] = data['Close_WPEA.PA'].rolling(window=10).mean()
data['MA30'] = data['Close_WPEA.PA'].rolling(window=30).mean()
data = data.rename(columns={'Close_WPEA.PA': 'price'})
data = data[['price', 'Volume_WPEA.PA', 'week_of_year', 'month_of_year', 'quarter_of_year', 'lag_1_week', 'Vol_1_month', 'MA10', 'MA30']]

# Step 3: Generate future dates
st.write("### Preparing Future Dates")
last_date = pd.Timestamp(data.index[-1])
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, num_days + 1)]
future_data = pd.DataFrame(index=future_dates)
future_data['price'] = np.nan
future_data['week_of_year'] = future_data.index.isocalendar().week
future_data['month_of_year'] = future_data.index.month
future_data['quarter_of_year'] = future_data.index.quarter

# Combine historical and future data
combined_data = pd.concat([data, future_data])
combined_data['lag_1_week'] = combined_data['price'].shift(7)
combined_data['Vol_1_month'] = combined_data['Volume_WPEA.PA'].shift(30)
combined_data['MA10'] = combined_data['price'].rolling(window=10).mean()
combined_data['MA30'] = combined_data['price'].rolling(window=30).mean()

# Step 4: Normalize the data
st.write("### Normalizing Data")
input_features = ['week_of_year', 'month_of_year', 'quarter_of_year', 'lag_1_week', 'Vol_1_month', 'MA10', 'MA30']
scaler.fit(combined_data[input_features].dropna())
ddf = pd.DataFrame(scaler.transform(combined_data[input_features].dropna()), 
                   columns=input_features, index=combined_data.dropna().index)

target_scaler.fit(combined_data[['price']].dropna())

# Step 5: Load the pre-trained model
st.write("### Loading Pre-Trained Model")
model = load_model('deploy/wpea_pred_model.h5', custom_objects={'mse': MeanSquaredError()})
model.compile(optimizer='adam', loss='mse')

# Step 6: Predict future prices
st.write("### Predicting Future Prices")
predictions_scaled = []
for date in future_dates:
    # Prepare input data
    last_sequence = ddf.loc[:date].iloc[-14:].values.reshape(1, 14, -1)  # Use the last 14 rows as input
    predicted_price = model.predict(last_sequence)[0][0]
    predictions_scaled.append(predicted_price)

    # Rescale the predicted price
    predicted_price_rescaled = target_scaler.inverse_transform([[predicted_price]])[0][0]
    combined_data.loc[date, 'price'] = predicted_price_rescaled

    # Recalculate features for the new row
    combined_data.loc[date, 'lag_1_week'] = combined_data['price'].shift(7).loc[date]
    combined_data.loc[date, 'Vol_1_month'] = combined_data['Volume_WPEA.PA'].shift(30).loc[date]
    combined_data.loc[date, 'MA10'] = combined_data['price'].rolling(window=10).mean().loc[date]
    combined_data.loc[date, 'MA30'] = combined_data['price'].rolling(window=30).mean().loc[date]

    # Scale the new row
    new_row = combined_data.loc[date, input_features].values.reshape(1, -1)
    new_row_scaled = scaler.transform(new_row)
    ddf = pd.concat([ddf, pd.DataFrame(new_row_scaled, columns=input_features, index=[date])])

# Step 7: Display results
st.write("### Results")
future_predictions = combined_data.loc[future_dates]
future_predictions = future_predictions.drop(columns=['Volume_WPEA.PA'])
st.dataframe(future_predictions)

# Download the dataset
csv = future_predictions.to_csv().encode('utf-8')
st.download_button(label="Download Predictions as CSV", data=csv, file_name='future_predictions.csv', mime='text/csv')

# Plot the price data
st.write("### Price Trend with Future Predictions")
historical_data = combined_data.loc[:last_date]
last_date = pd.Timestamp(data.index[-2])
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, num_days + 1)]
predicted_data = combined_data.loc[future_dates]

fig = go.Figure()
fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['price'], mode='lines', name='Historical', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['price'], mode='lines', name='Predicted', line=dict(color='red')))
fig.update_layout(title='Price Trend with Future Predictions', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig)