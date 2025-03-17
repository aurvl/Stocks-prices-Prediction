import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
from rsier import RSI
import pickle

# Load the MinMaxScaler objects
with open('src/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

scaler, target_scaler = preprocessor

# Step 0: Download historical data
print("\nPart 1: Download historical data")
symbol = "CW8.PA"
data = yf.download(symbol)
data.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in data.columns]
data['dat'] = data.index
data = data[data['dat'] > '2018-04-17'] # remove data before 2018-04-17
print("part 1 done!\n")

# Step 1: Add feature columns to historical data
print("Part 2: Add feature columns to historical data")
data['day_of_week'] = data['dat'].dt.dayofweek
data['week_of_year'] = data['dat'].dt.isocalendar().week
data['month_of_year'] = data['dat'].dt.month
data['quarter_of_year'] = data['dat'].dt.quarter
data['semester_of_year'] = data['dat'].dt.quarter.apply(lambda x: 1 if x < 3 else 2)

data = data.rename(columns={'Close_CW8.PA': 'price', 'Volume_CW8.PA': 'Volume'})
data = data[['price', 'Volume', 'day_of_week', 'week_of_year', 'month_of_year', 'quarter_of_year', 'semester_of_year']]
print("part 2 done!\n")

# Step 2: Identify the last date and generate future dates
print("Part 3: Identify the last date and generate future dates")
last_date = pd.Timestamp(data.index[-1])  # Ensure last_date is a Timestamp object
future_dates = []
i = 1
while len(future_dates) < 20:
    next_date = last_date + pd.Timedelta(days=i)
    if next_date.weekday() < 5:  # exclut Samedi = 5 et Dimanche = 6
        future_dates.append(next_date)
    i += 1
future_data = pd.DataFrame(index=future_dates)
print("part 3 done!\n")

# Step 3: Add feature columns to the future data
print("Part 4: Add feature columns to the future data")
future_data['price'] = np.nan
future_data['day_of_week'] = future_data.index.dayofweek
future_data['week_of_year'] = future_data.index.isocalendar().week
future_data['month_of_year'] = future_data.index.month
future_data['quarter_of_year'] = future_data.index.quarter
future_data['semester_of_year'] = future_data['quarter_of_year'].apply(lambda x: 1 if x < 3 else 2)

# Combine historical and future data
combined_data = pd.concat([data, future_data])

# Calculate additional features for the combined data
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
combined_data.to_csv('deploy/combined_data.csv')
print("part 4 done!\n")

# Step 4: Normalize the data
print("Part 5: Normalize the data")
input_features = ['day_of_week', 'week_of_year', 'month_of_year', 'quarter_of_year', 'semester_of_year', 
                  'lag_1_week', 'Vol_1_month', 'SMA20', 'SMA50', 'RSI', 'return']  # Exclude 'price'
ddf = pd.DataFrame(scaler.transform(combined_data[input_features].dropna()), 
                   columns=input_features, index=combined_data.dropna().index)

# Step 5: Load the pre-trained model
model = load_model('src/CW8_pred_model.h5', custom_objects={'mse': MeanSquaredError()})
model.compile(optimizer='adam', loss='mse')
print("part 5 done!\n")

# Predict the price for the next 30 days sequentially
print("Part 6: Predict the price for the next 30 days sequentially")
headers = "date, predicted_value, price\n"
file_path = 'deploy/predicts.txt'

with open(file_path, 'w') as f:  
    f.write(headers)

predictions_scaled = []
for date in future_dates:
    # Prepare input data
    last_sequence = ddf.loc[:date].iloc[-10:].values.reshape(1, 10, -1)  # Use the last 14 rows as input
    print(last_sequence.shape)
    print("Check1: input data prepared")

    # Predict the next price
    predicted_price = model.predict(last_sequence)[0][0]
    predictions_scaled.append(predicted_price)
    print("Check 2: prediction done")

    # Inverse transform the scaled prediction to original price
    prediction_rescaled = target_scaler.inverse_transform([[predicted_price]])[0][0]
    
    with open(file_path, 'a') as f:
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
    print("Check 3: new row scaled")
    new_row = combined_data.loc[date, input_features].values.reshape(1, -1)
    new_row_scaled = scaler.transform(new_row)
    print("Next row :", new_row_scaled)
    # Update ddf with the scaled new row
    ddf = pd.concat([ddf, pd.DataFrame(new_row_scaled, columns=input_features, index=[date])])
    print("Check 4: ddf updated\n")


print("Part 7: Display the DataFrame for the future predictions")
# Display the DataFrame for the future predictions
combined_data = combined_data.drop(columns=['Volume'])
future_predictions = combined_data.loc[future_dates]

# Plot the price data
plt.figure(figsize=(12, 6))
plt.plot(combined_data.index, combined_data['price'], label='Price', color='#0072B2')
plt.fill_between(combined_data.index, combined_data['price'], color="#0072B2", alpha=0.3)
plt.title('Price Trend with Future Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.axvline(x=last_date, color='red', linestyle='--', linewidth=1, label='Prediction Start')
plt.legend()
plt.grid(alpha=0.3)
plt.show()