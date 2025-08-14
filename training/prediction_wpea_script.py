import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from functions import model_builder, RSI
import pickle


symbol = "CW8.PA"
data = yf.download(symbol)

data.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in data.columns]
data['dat'] = data.index
data = data[data['dat'] > '2018-04-17'] # remove data before 2018-04-17

# Add features
data['day_of_week'] = data['dat'].dt.dayofweek
data['week_of_year'] = data['dat'].dt.isocalendar().week
data['month_of_year'] = data['dat'].dt.month
data['quarter_of_year'] = data['dat'].dt.quarter
data['semester_of_year'] = data['dat'].dt.quarter.apply(lambda x: 1 if x < 3 else 2)

# Add the lag feature (value of the index 1 week prior)
data['lag_1_week'] = data['Close_CW8.PA'].shift(5)
data['Vol_1_month'] = data['Volume_CW8.PA'].shift(20)

data['SMA20'] = data['Close_CW8.PA'].rolling(window=20).mean()
data['SMA50'] = data['Close_CW8.PA'].rolling(window=50).mean()
data['RSI'] = RSI(data['Close_CW8.PA'])
data['return'] = data['Close_CW8.PA'].pct_change()

# Drop rows with NaN (due to lag feature)
data = data.dropna(subset='SMA50')

df = data[['Close_CW8.PA', 'day_of_week', 'week_of_year', 'month_of_year', 'quarter_of_year', 
           'semester_of_year', 'lag_1_week', 'Vol_1_month', 'SMA20', 'SMA50', 'RSI', 'return']]
df = df.rename(columns={'Close_CW8.PA':'price'})

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

sequence_length = 10 # 2 semaines (ya pas samdi et dimanche)
X, y = create_sequences(ddf, target_column="price", sequence_length=sequence_length)

# Spliting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape=(X_train.shape[1], X_train.shape[2])
model = model_builder(input_shape)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)

# History plot
train_loss = history.history['loss']
val_loss = history.history['val_loss']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

## Plot Train Loss
axes[0].plot(range(1, len(train_loss) + 1), train_loss, marker='o', color='blue')
axes[0].set_title('Train Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True)

## Plot Validation Loss
axes[1].plot(range(1, len(val_loss) + 1), val_loss, marker='o', color='orange')
axes[1].set_title('Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('loss_plots.png', dpi=300)
plt.close(fig)

# Prédictions
predictions = model.predict(X_test)

target_scaler = MinMaxScaler()
target_scaler.fit_transform(df[['price']])
scaler.fit_transform(df.drop(columns=['price']))

preprocessors = (scaler, target_scaler)
with open('src/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessors, f)

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

model.save("src/CW8_pred_model.h5")
print("\n✅ Model saved successfully!")