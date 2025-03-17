import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from keras.losses import MeanSquaredError
import plotly.graph_objects as go
import pickle
from rsier import RSI


with open('src/preprocessor.pkl', 'rb') as f:
    preprocessors = pickle.load(f)

scaler, target_scaler = preprocessors

# Streamlit app title
st.title("Stock Price Prediction")
st.write("This application predicts the future stock prices for [CW8.PA](https://finance.yahoo.com/quote/CW8.PA/) using a pre-trained LSTM model.")
st.write("Lien vers le repository : [GitHub](https://github.com/aurvl/Stocks-prices-Prediction)")

# Step 0: User Input
st.sidebar.header("Prediction Settings")
num_days = st.sidebar.slider("Number of days to predict:", min_value=1, max_value=20, value=20)
st.sidebar.write("Predicting for:", num_days, "days")

# Step 1: Download historical data
st.write("### Downloading Historical Data")
symbol = "CW8.PA"
data = yf.download(symbol)
data.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in data.columns]
data['dat'] = data.index
data = data[data['dat'] > '2018-04-17'] # remove data before 2018-04-17

# Step 2: Add feature columns to historical data
st.write("### Adding Feature Columns")
data['day_of_week'] = data['dat'].dt.dayofweek
data['week_of_year'] = data['dat'].dt.isocalendar().week
data['month_of_year'] = data['dat'].dt.month
data['quarter_of_year'] = data['dat'].dt.quarter
data['semester_of_year'] = data['dat'].dt.quarter.apply(lambda x: 1 if x < 3 else 2)
data = data.rename(columns={'Close_CW8.PA': 'price', 'Volume_CW8.PA': 'Volume'})
data = data[['price', 'Volume', 'day_of_week', 'week_of_year', 'month_of_year', 'quarter_of_year', 'semester_of_year']]

# Step 3: Generate future dates
st.write("### Preparing Future Dates")
last_date = pd.Timestamp(data.index[-1])  # Ensure last_date is a Timestamp object
future_dates = []
i = 1
while len(future_dates) < 20:
    next_date = last_date + pd.Timedelta(days=i)
    if next_date.weekday() < 5:  # exclut Samedi = 5 et Dimanche = 6
        future_dates.append(next_date)
    i += 1
future_data = pd.DataFrame(index=future_dates)
future_data['price'] = np.nan
future_data['day_of_week'] = future_data.index.dayofweek
future_data['week_of_year'] = future_data.index.isocalendar().week
future_data['month_of_year'] = future_data.index.month
future_data['quarter_of_year'] = future_data.index.quarter
future_data['semester_of_year'] = future_data['quarter_of_year'].apply(lambda x: 1 if x < 3 else 2)

# Combine historical and future data
combined_data = pd.concat([data, future_data])
combined_data['lag_1_week'] = combined_data['price'].shift(5)
combined_data['Vol_1_month'] = combined_data['Volume'].shift(20)
combined_data['SMA20'] = combined_data['price'].rolling(window=20).mean()
combined_data['SMA50'] = combined_data['price'].rolling(window=50).mean()
combined_data['RSI'] = RSI(combined_data['price'])
combined_data['return'] = combined_data['price'].pct_change()
combined_data = combined_data[['price', 'Volume', 'day_of_week', 'week_of_year', 'month_of_year', 
                               'quarter_of_year', 'semester_of_year', 'lag_1_week', 'Vol_1_month', 
                               'SMA20', 'SMA50', 'RSI', 'return']]
combined_data = combined_data.iloc[49:]

# Step 4: Normalize the data
st.write("### Normalizing Data")
input_features = ['day_of_week', 'week_of_year', 'month_of_year', 'quarter_of_year', 'semester_of_year', 
                  'lag_1_week', 'Vol_1_month', 'SMA20', 'SMA50', 'RSI', 'return']  # Exclude 'price'
ddf = pd.DataFrame(scaler.transform(combined_data[input_features].dropna()), 
                   columns=input_features, index=combined_data.dropna().index)

# Step 5: Load the pre-trained model
st.write("### Loading Pre-Trained Model")
model = load_model('src/CW8_pred_model.h5', custom_objects={'mse': MeanSquaredError()})
model.compile(optimizer='adam', loss='mse')

# Step 6: Predict future prices
st.write("### Predicting Future Prices")
predictions_scaled = []
for date in future_dates:
    # Prepare input data
    last_sequence = ddf.loc[:date].iloc[-10:].values.reshape(1, 10, -1)  # Use the last 14 rows as input

    # Predict the next price
    predicted_price = model.predict(last_sequence)[0][0]
    predictions_scaled.append(predicted_price)

    # Inverse transform the scaled prediction to original price
    prediction_rescaled = target_scaler.inverse_transform([[predicted_price]])[0][0]
    with open('deploy/predicts.txt', 'a') as f:
        f.write(f"{date}, {predicted_price}, {prediction_rescaled}\n")

    # Add rescaled predicted price to combined_data
    combined_data.loc[date, 'price'] = prediction_rescaled

    # Recalculate features for the new row in combined_data
    combined_data.loc[date, 'lag_1_week'] = combined_data['price'].shift(7).loc[date]
    combined_data.loc[date, 'Vol_1_month'] = combined_data['Volume'].shift(20).loc[date]
    combined_data.loc[date, 'SMA20'] = combined_data['price'].rolling(window=20).mean().loc[date]
    combined_data.loc[date, 'SMA50'] = combined_data['price'].rolling(window=50).mean().loc[date]
    combined_data['RSI'] = RSI(combined_data['price'])
    combined_data.loc[date, 'return'] = combined_data['price'].pct_change().loc[date]

    # Scaling new row
    new_row = combined_data.loc[date, input_features].values.reshape(1, -1)
    new_row_scaled = scaler.transform(new_row)
    ddf = pd.concat([ddf, pd.DataFrame(new_row_scaled, columns=input_features, index=[date])])

# Step 7: Display results
st.write("### Results")
combined_data = combined_data.drop(columns=['Volume'])
future_predictions = combined_data.loc[future_dates]
st.dataframe(future_predictions)

# Download the dataset
csv = future_predictions.to_csv().encode('utf-8')
today = pd.Timestamp.today().strftime('%Y-%m-%d')
st.download_button(label="Download Predictions as CSV", data=csv, file_name=f'predictions_on_{today}.csv', 
                   mime='text/csv')

# Plot the price data
st.write("### Price Trend with Future Predictions")
historical_data = combined_data.loc[:last_date]
last_date = pd.Timestamp(data.index[-1])
future_dates = [last_date] + future_dates
predicted_data = combined_data.loc[future_dates]

# color
last_real_price = historical_data['price'].iloc[-1]
last_predicted_price = predicted_data['price'].iloc[-1]

if last_predicted_price > last_real_price:
    pred_color = 'green'
    pred_fillcolor = 'rgba(0, 255, 0, 0.3)'
else:
    pred_color = 'red'
    pred_fillcolor = 'rgba(255, 0, 0, 0.3)'

fig = go.Figure()
fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['price'], mode='lines', 
                         name='Historical', line=dict(color='blue'), fill='tozeroy',
                         fillcolor='rgba(0,100,80,0.2)'))
fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['price'], mode='lines', 
                         name='Predicted', line=dict(color=pred_color), fill='tozeroy',
                         fillcolor=pred_fillcolor))
fig.update_layout(title='Price Trend with Future Predictions', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig)


# Over 2 last months
today = pd.Timestamp.today()
two_months_ago = today - pd.DateOffset(months=2)
two_month_data = historical_data.loc[historical_data.index >= two_months_ago]

fig = go.Figure()
fig.add_trace(go.Scatter(x=two_month_data.index, y=two_month_data['price'], mode='lines', 
                         name='Historical', line=dict(color='blue'), fill='tozeroy',
                         fillcolor='rgba(0,100,80,0.2)'))
fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['price'], mode='lines', 
                         name='Predicted', line=dict(color=pred_color), fill='tozeroy',
                         fillcolor=pred_fillcolor))
fig.update_layout(title='ON LAST 2 MONTHS', xaxis_title='Date', yaxis_title='Price',
                  yaxis=dict(range=[min(two_month_data['price']) * 0.95, max(two_month_data['price']) * 1.05]))
st.plotly_chart(fig)

st.write("Contact : [Aurel VEHI](https://github.com/aurvl)")
