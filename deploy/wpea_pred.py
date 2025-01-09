import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
import matplotlib.pyplot as plt

# Step 0: Download historical data
print("Part 1")
symbol = "WPEA.PA"
data = yf.download(symbol)
data.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in data.columns]
data = data[['Close_WPEA.PA', 'Volume_WPEA.PA']]
data['dat'] = data.index

# Step 1: Add feature columns to historical data
print("Part 2")
data['week_of_year'] = data['dat'].dt.isocalendar().week
data['month_of_year'] = data['dat'].dt.month
data['quarter_of_year'] = data['dat'].dt.quarter
data['lag_1_week'] = data['Close_WPEA.PA'].shift(7)
data['Vol_1_month'] = data['Volume_WPEA.PA'].shift(30)
data['MA10'] = data['Close_WPEA.PA'].rolling(window=10).mean()
data['MA30'] = data['Close_WPEA.PA'].rolling(window=30).mean()
data = data.rename(columns={'Close_WPEA.PA': 'price'})
data = data[['price', 'Volume_WPEA.PA', 'week_of_year', 'month_of_year', 'quarter_of_year', 'lag_1_week', 'Vol_1_month', 'MA10', 'MA30']]
data = data.dropna()

# Step 2: Identify the last date and generate future dates
print("Part 3")
last_date = pd.Timestamp(data.index[-1])  # Ensure last_date is a Timestamp object
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]
future_data = pd.DataFrame(index=future_dates)

# Step 3: Add feature columns to the future data
print("Part 4")
future_data['price'] = np.nan
future_data['week_of_year'] = future_data.index.isocalendar().week
future_data['month_of_year'] = future_data.index.month
future_data['quarter_of_year'] = future_data.index.quarter

# Combine historical and future data
combined_data = pd.concat([data, future_data])

# Calculate additional features for the combined data
combined_data['lag_1_week'] = combined_data['price'].shift(7)
combined_data['Vol_1_month'] = combined_data['Volume_WPEA.PA'].shift(30)
combined_data['MA10'] = combined_data['price'].rolling(window=10).mean()
combined_data['MA30'] = combined_data['price'].rolling(window=30).mean()
combined_data = combined_data[['price', 'week_of_year', 'month_of_year', 'quarter_of_year', 'lag_1_week', 'Vol_1_month', 'MA10', 'MA30']]

# Step 4: Normalize the data
print("Part 5")
scaler = MinMaxScaler()
input_features = ['week_of_year', 'month_of_year', 'quarter_of_year', 'lag_1_week', 'Vol_1_month', 'MA10', 'MA30']  # Exclude 'price'
scaler.fit(combined_data[input_features].dropna())  # Fit only on input features (7 columns)
ddf = pd.DataFrame(scaler.transform(combined_data[input_features].dropna()), 
                   columns=input_features, index=combined_data.dropna().index)

# Prepare the target scaler outside the loop
target_scaler = MinMaxScaler()
target_scaler.fit(combined_data[['price']])

# Step 5: Load the pre-trained model
model = load_model('wpea_pred_model.h5', custom_objects={'mse': MeanSquaredError()})
model.compile(optimizer='adam', loss='mse')

# Predict the price for the next 30 days sequentially
print("Part 6")
predictions_scaled = []
for date in future_dates:
    # Prepare input data
    last_sequence = ddf.loc[:date].iloc[-14:].values.reshape(1, 14, -1)  # Use the last 14 rows as input
    print(last_sequence.shape)
    print("Check 1")

    # Predict the next price
    predicted_price = model.predict(last_sequence)[0][0]
    predictions_scaled.append(predicted_price)
    print("Check 2")

    # Inverse transform the scaled prediction to original price
    predicted_price_rescaled = target_scaler.inverse_transform([[predicted_price]])[0][0]

    # Add rescaled predicted price to combined_data
    combined_data.loc[date, 'price'] = predicted_price_rescaled

    # Recalculate features for the new row in combined_data
    combined_data.loc[date, 'lag_1_week'] = combined_data['price'].shift(7).loc[date]
    combined_data.loc[date, 'Vol_1_month'] = combined_data['Vol_1_month'].shift(30).loc[date]
    combined_data.loc[date, 'MA10'] = combined_data['price'].rolling(window=10).mean().loc[date]
    combined_data.loc[date, 'MA30'] = combined_data['price'].rolling(window=30).mean().loc[date]

    # Scale the new row
    print("Check 3")
    new_row = combined_data.loc[date, input_features].values.reshape(1, -1)
    new_row_scaled = scaler.transform(new_row)
    print(new_row_scaled)
    # Update ddf with the scaled new row
    ddf = pd.concat([ddf, pd.DataFrame(new_row_scaled, columns=input_features, index=[date])])
    print("Check 4")


print("Part 7")
# Rescale all predictions after the loop
predictions_rescaled = target_scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

# Update combined_data with rescaled predictions
combined_data.loc[future_dates, 'price'] = predictions_rescaled

# Display the DataFrame for the future predictions
future_predictions = combined_data.loc[future_dates]

# Plot the price data
plt.figure(figsize=(12, 6))
plt.plot(combined_data.index, combined_data['price'], label='Price', color='blue')
plt.title('Price Trend with Future Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.axvline(x=last_date, color='red', linestyle='--', label='Prediction Start')
plt.legend()
plt.grid()
plt.show()