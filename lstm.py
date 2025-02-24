import numpy as np
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load NetCDF Data
file_path = "D:/IMDrainfall_1901-2023.nc"  # Your NetCDF file path
ds = xr.open_dataset(file_path, engine="netcdf4")

# Select a specific LATITUDE and LONGITUDE (e.g., Central India)
selected_lat = 20.0  # Adjust as needed
selected_lon = 80.0  # Adjust as needed

# Find nearest latitude and longitude indices
lat_idx = np.abs(ds.LATITUDE - selected_lat).argmin()
lon_idx = np.abs(ds.LONGITUDE - selected_lon).argmin()

# Extract rainfall time series at the chosen location
rainfall_data = ds["RAINFALL"][:, lat_idx, lon_idx].values  # Shape: (43829,)
time_data = ds["TIME"].values  # Dates

# Normalize the data (Min-Max Scaling)
rainfall_min, rainfall_max = np.min(rainfall_data), np.max(rainfall_data)
rainfall_data = (rainfall_data - rainfall_min) / (rainfall_max - rainfall_min)

# Close dataset to free memory
ds.close()

# Prepare Data for LSTM
def prepare_sequences(data, seq_length=50):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        Y.append(data[i + seq_length])
    return np.array(X), np.array(Y)

seq_length = 50  # Number of past days to consider for prediction
X, Y = prepare_sequences(rainfall_data, seq_length)

# Reshape for LSTM (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))
Y = Y.reshape((Y.shape[0], 1))

# Define and Compile LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the Model
history = model.fit(X, Y, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

# Plot Training Loss
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Make Predictions
test_sample = X[-1].reshape(1, seq_length, 1)  # Last sequence for testing
predicted_rainfall = model.predict(test_sample)

# Convert back to original scale
predicted_rainfall = predicted_rainfall * (rainfall_max - rainfall_min) + rainfall_min

print(f"Predicted Rainfall at Lat {selected_lat}, Lon {selected_lon}: {predicted_rainfall[0][0]:.2f} mm")
