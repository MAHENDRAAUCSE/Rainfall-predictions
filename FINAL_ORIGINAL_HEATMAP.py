import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import cartopy.crs as ccrs
from PIL import Image
import matplotlib.gridspec as gridspec

# File path for the dataset
file_path = "D:/IMDrainfall_1901-2023.nc"

# Load the dataset with xarray (using chunking)
ds = xr.open_dataset(file_path, chunks={'TIME': 1000})

# Define boundaries for Andhra Pradesh (Region of Interest)
andhra_lon_min, andhra_lon_max = 76.0, 84.0
andhra_lat_min, andhra_lat_max = 12.0, 20.0

# Extract data for Andhra Pradesh
ds_andhra = ds.sel(LONGITUDE=slice(andhra_lon_min, andhra_lon_max),
                   LATITUDE=slice(andhra_lat_min, andhra_lat_max))

# Split data into train, validation, and test sets based on time
train_data = ds_andhra.sel(TIME=slice("1901-01-01", "1990-12-31"))
test_data = ds_andhra.sel(TIME=slice("1991-01-01", "2011-12-31"))
val_data= ds_andhra.sel(TIME=slice("2012-01-01", "2023-12-31"))

# Load rainfall data and compute values
rainfall_train = train_data["RAINFALL"].compute()
rainfall_test = test_data["RAINFALL"].compute()
rainfall_val = val_data["RAINFALL"].compute()


# Downsample spatially (select every second grid point)
rainfall_train = rainfall_train[:, ::2, ::2]
rainfall_test = rainfall_test[:, ::2, ::2]
rainfall_val = rainfall_val[:, ::2, ::2]


# Fill missing values with the mean of each dataset
rainfall_train = rainfall_train.fillna(rainfall_train.mean())
rainfall_test = rainfall_test.fillna(rainfall_test.mean())
rainfall_val = rainfall_val.fillna(rainfall_val.mean())


# Determine spatial dimensions and flatten the 2D grid into 1D features
n_lat = rainfall_train.shape[1]
n_lon = rainfall_train.shape[2]
n_features = n_lat * n_lon

rainfall_train = rainfall_train.values.reshape(-1, n_features).astype(np.float32)
rainfall_test = rainfall_test.values.reshape(-1, n_features).astype(np.float32)
rainfall_val = rainfall_val.values.reshape(-1, n_features).astype(np.float32)


# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
rainfall_train = scaler.fit_transform(rainfall_train)
rainfall_test = scaler.transform(rainfall_test)
rainfall_val = scaler.transform(rainfall_val)


class RainfallGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, time_steps, batch_size):
        self.data = data
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.n_samples = len(data) - time_steps
        self.indices = np.arange(self.n_samples)

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_indices = self.indices[start:end]
        X_batch = np.zeros((len(batch_indices), self.time_steps, self.data.shape[1]), dtype=np.float32)
        y_batch = np.zeros((len(batch_indices), self.data.shape[1]), dtype=np.float32)
        for i, j in enumerate(batch_indices):
            X_batch[i] = self.data[j:j + self.time_steps]
            y_batch[i] = self.data[j + self.time_steps]
        return X_batch, y_batch

# Set time_steps and batch_size
time_steps = 30
batch_size = 32

# Pass the preprocessed NumPy arrays (not the xarray objects) to the generator
train_gen = RainfallGenerator(rainfall_train, time_steps, batch_size)
test_gen = RainfallGenerator(rainfall_test, time_steps, batch_size)
val_gen = RainfallGenerator(rainfall_val, time_steps, batch_size)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(time_steps, n_features)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(n_features)
])

model.compile(optimizer='adam', loss='mse')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(train_gen, epochs=20, validation_data=val_gen, callbacks=[early_stopping], verbose=1)

test_predict = model.predict(test_gen, steps=len(test_gen))
test_predict = scaler.inverse_transform(test_predict)

# Get actual test targets (starting at index time_steps)
test_actual = test_gen.data[time_steps:]
test_actual = scaler.inverse_transform(test_actual)

# Reshape to spatial dimensions (time, n_lat, n_lon)
test_predict_reshaped = test_predict.reshape(-1, n_lat, n_lon)
test_actual_reshaped = test_actual.reshape(-1, n_lat, n_lon)

# Predict on validation data
val_predict = model.predict(val_gen, steps=len(val_gen))
val_predict = scaler.inverse_transform(val_predict)

val_actual = val_gen.data[time_steps:]
val_actual = scaler.inverse_transform(val_actual)

val_predict_reshaped = val_predict.reshape(-1, n_lat, n_lon)
val_actual_reshaped = val_actual.reshape(-1, n_lat, n_lon)

# For time-series error calculation, flatten the arrays
test_actual_flat = test_actual_reshaped.flatten()
test_predict_flat = test_predict_reshaped.flatten()
val_actual_flat = val_actual_reshaped.flatten()
val_predict_flat = val_predict_reshaped.flatten()
# Set fixed contour levels from 1.0 to 6.0
fixed_levels = np.linspace(1.0, 6.0, 20)

# Path to the map image
map_path = "C:/Users/Mahen/AppData/Local/Programs/Python/Python311/Lib/site-packages/andhra-pradesh-blank-map.jpg"
map_img = Image.open(map_path)

# plotting
fig = plt.figure(figsize=(18, 18))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8])
ax1 = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax1.imshow(map_img, extent=[andhra_lon_min, andhra_lon_max, andhra_lat_min, andhra_lat_max], alpha=0.6)
contour1 = ax1.contourf(ds_andhra.LONGITUDE.values[::2],
                        ds_andhra.LATITUDE.values[::2],
                        test_actual_reshaped.mean(axis=0),  # average over time steps
                        cmap='coolwarm', alpha=0.8, levels=fixed_levels)
plt.colorbar(contour1, ax=ax1, label='Rainfall (mm)')
ax1.set_title("Test Set: Actual Rainfall (Avg)")
ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax2.imshow(map_img, extent=[andhra_lon_min, andhra_lon_max, andhra_lat_min, andhra_lat_max], alpha=0.6)
contour2 = ax2.contourf(ds_andhra.LONGITUDE.values[::2],
                        ds_andhra.LATITUDE.values[::2],
                        test_predict_reshaped.mean(axis=0),  # average over time steps
                        cmap='coolwarm', alpha=0.8, levels=fixed_levels)
plt.colorbar(contour2, ax=ax2, label='Rainfall (mm)')
ax2.set_title("Test Set: Predicted Rainfall (Avg)")
ax3 = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax3.imshow(map_img, extent=[andhra_lon_min, andhra_lon_max, andhra_lat_min, andhra_lat_max], alpha=0.6)
contour3 = ax3.contourf(ds_andhra.LONGITUDE.values[::2],
                        ds_andhra.LATITUDE.values[::2],
                        val_actual_reshaped.mean(axis=0),  # average over time steps
                        cmap='coolwarm', alpha=0.8, levels=fixed_levels)
plt.colorbar(contour3, ax=ax3, label='Rainfall (mm)')
ax3.set_title("Validation Set: Actual Rainfall (Avg)")
ax4 = plt.subplot(gs[1, 1], projection=ccrs.PlateCarree())
ax4.imshow(map_img, extent=[andhra_lon_min, andhra_lon_max, andhra_lat_min, andhra_lat_max], alpha=0.6)
contour4 = ax4.contourf(ds_andhra.LONGITUDE.values[::2],
                        ds_andhra.LATITUDE.values[::2],
                        val_predict_reshaped.mean(axis=0),  # average over time steps
                        cmap='coolwarm', alpha=0.8, levels=fixed_levels)
plt.colorbar(contour4, ax=ax4, label='Rainfall (mm)')
ax4.set_title("Validation Set: Predicted Rainfall (Avg)")
plt.tight_layout()
plt.show()

mse_test = mean_squared_error(test_actual_flat, test_predict_flat)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(test_actual_flat, test_predict_flat)
print(f"Test Set - MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, R²: {r2_test:.2f}")

mse_val = mean_squared_error(val_actual_flat, val_predict_flat)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(val_actual_flat, val_predict_flat)
print(f"Validation Set - MSE: {mse_val:.2f}, RMSE: {rmse_val:.2f}, R²: {r2_val:.2f}")