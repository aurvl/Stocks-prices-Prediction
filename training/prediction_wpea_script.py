import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

symbol = "WPEA.PA"
data = yf.download(symbol)

data.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in data.columns]
data['dat'] = data.index

data['week_of_year'] = data['dat'].dt.isocalendar().week
data['month_of_year'] = data['dat'].dt.month
data['quarter_of_year'] = data['dat'].dt.quarter

# Add the lag feature (value of the index 1 week prior)
data['lag_1_week'] = data['Close_WPEA.PA'].shift(7)
data['Vol_1_month'] = data['Volume_WPEA.PA'].shift(30)

data['MA10'] = data['Close_WPEA.PA'].rolling(window=10).mean()
data['MA30'] = data['Close_WPEA.PA'].rolling(window=30).mean()

# Drop rows with NaN (due to lag feature)
data = data.dropna()

df = data[['Close_WPEA.PA', 'week_of_year', 'month_of_year', 'quarter_of_year', 'lag_1_week', 'Vol_1_month', 'MA10', 'MA30']]
df = df.rename(columns={'Close_WPEA.PA':'Price'})

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)
ddf = pd.DataFrame(normalized_data, columns=df.columns)

def create_sequences(data, target_column, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        seq_x = data.iloc[i:i + sequence_length].drop(columns=[target_column]).values
        seq_y = data.iloc[i + sequence_length][target_column]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

sequence_length = 14 # 2 semaines
X, y = create_sequences(ddf, target_column="Price", sequence_length=sequence_length)

# Spliting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)  # Output layer for predicting a single value
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)

# Prédictions
predictions = model.predict(X_test)

target_scaler = MinMaxScaler()
target_scaler.fit_transform(df[['Price']])

# Comparer les valeurs réelles et prédites
predictions = predictions.reshape(-1, 1)
predictions_rescaled = target_scaler.inverse_transform(predictions)
y_test_scaled = y_test.reshape(-1, 1)
y_test_rescaled = target_scaler.inverse_transform(y_test_scaled)

results = pd.DataFrame({
    'Actual': y_test_rescaled.flatten(),
    'Predicted': predictions_rescaled.flatten()
})

results.to_excel('src/results.xlsx', sheet_name='Actual vs Predicted')

mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_rescaled, predictions_rescaled)

with open('src/metrics.txt', 'w') as f:
    f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")

model.save("src/wpea_pred_model.h5")