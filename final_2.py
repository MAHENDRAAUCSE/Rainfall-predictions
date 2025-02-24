import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

file_path = "D:/IMDrainfall_1901-2023.nc"
ds = xr.open_dataset(file_path, chunks={'TIME': 1000})
andhra_lon_min, andhra_lon_max = 76.0, 84.0
andhra_lat_min, andhra_lat_max = 12.0, 20.0
ds_andhra = ds.sel(LONGITUDE=slice(andhra_lon_min, andhra_lon_max), LATITUDE=slice(andhra_lat_min, andhra_lat_max))
train_data = ds_andhra.sel(TIME=slice("1901-01-01", "1990-12-31"))
val_data = ds_andhra.sel(TIME=slice("1991-01-01", "2011-12-31"))
test_data = ds_andhra.sel(TIME=slice("2012-01-01", "2023-12-31"))
rainfall_train = train_data["RAINFALL"].compute()
rainfall_val = val_data["RAINFALL"].compute()
rainfall_test = test_data["RAINFALL"].compute()
rainfall_train = rainfall_train[:, ::2, ::2]
rainfall_val = rainfall_val[:, ::2, ::2]
rainfall_test = rainfall_test[:, ::2, ::2]

from scipy.interpolate import griddata

def interpolate_missing(data):
    lon, lat = np.meshgrid(data.LONGITUDE.values, data.LATITUDE.values)
    filled_data = np.array([
        griddata(
            points=(lon.flatten(), lat.flatten()),
            values=data[t].values.flatten(),
            xi=(lon, lat),
            method='linear'
        )
        for t in range(len(data.TIME))
    ])
    filled_data = np.nan_to_num(filled_data, nan=np.nanmean(filled_data))
    return xr.DataArray(filled_data, dims=('TIME', 'LATITUDE', 'LONGITUDE'), coords=data.coords)


rainfall_train = interpolate_missing(rainfall_train)
rainfall_val = interpolate_missing(rainfall_val)
rainfall_test = interpolate_missing(rainfall_test)
n_lat = rainfall_train.shape[1]
n_lon = rainfall_train.shape[2]
n_features = n_lat * n_lon
rainfall_train = rainfall_train.values.reshape(-1, n_features).astype(np.float32)
rainfall_val = rainfall_val.values.reshape(-1, n_features).astype(np.float32)
rainfall_test = rainfall_test.values.reshape(-1, n_features).astype(np.float32)

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
rainfall_train = scaler.fit_transform(rainfall_train)
rainfall_val = scaler.transform(rainfall_val)
rainfall_test = scaler.transform(rainfall_test)


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
time_steps = 90
batch_size = 64

train_gen = RainfallGenerator(rainfall_train, time_steps, batch_size)
val_gen = RainfallGenerator(rainfall_val, time_steps, batch_size)
test_gen = RainfallGenerator(rainfall_test, time_steps, batch_size)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(time_steps, n_features), kernel_regularizer='l2'),
    Dropout(0.3),
    LSTM(128, return_sequences=True, kernel_regularizer='l2'),
    Dropout(0.3),
    LSTM(64, kernel_regularizer='l2'),
    Dropout(0.3),
    Dense(n_features)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
history = model.fit(train_gen, epochs=50, validation_data=val_gen,  callbacks=[early_stopping, reduce_lr], verbose=1)
test_predict = model.predict(test_gen, steps=len(test_gen))
test_predict = scaler.inverse_transform(test_predict)
test_actual = test_gen.data[time_steps:]
test_actual = scaler.inverse_transform(test_actual)
val_predict = model.predict(val_gen, steps=len(val_gen))
val_predict = scaler.inverse_transform(val_predict)
val_actual = val_gen.data[time_steps:]
val_actual = scaler.inverse_transform(val_actual)
test_predict_reshaped = test_predict.reshape(-1, n_lat, n_lon)
test_actual_reshaped = test_actual.reshape(-1, n_lat, n_lon)
val_predict_reshaped = val_predict.reshape(-1, n_lat, n_lon)
val_actual_reshaped = val_actual.reshape(-1, n_lat, n_lon)
test_actual_flat = test_actual.flatten()
test_predict_flat = test_predict.flatten()
val_actual_flat = val_actual.flatten()
val_predict_flat = val_predict.flatten()


if np.any(np.isnan(test_actual_flat)) or np.any(np.isnan(test_predict_flat)):
    print("NaN values detected in test data!")
    test_actual_flat = np.nan_to_num(test_actual_flat, nan=0.0)
    test_predict_flat = np.nan_to_num(test_predict_flat, nan=0.0)
if np.any(np.isnan(val_actual_flat)) or np.any(np.isnan(val_predict_flat)):
    print("NaN values detected in validation data!")
    val_actual_flat = np.nan_to_num(val_actual_flat, nan=0.0)
    val_predict_flat = np.nan_to_num(val_predict_flat, nan=0.0)

from sklearn.metrics import mean_squared_error, r2_score

mse_test = mean_squared_error(test_actual_flat, test_predict_flat)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(test_actual_flat, test_predict_flat)
mse_val = mean_squared_error(val_actual_flat, val_predict_flat)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(val_actual_flat, val_predict_flat)
print(f"Test Set - MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, R²: {r2_test:.2f} ({r2_test * 100:.2f}%)")
print(f"Validation Set - MSE: {mse_val:.2f}, RMSE: {rmse_val:.2f}, R²: {r2_val:.2f} ({r2_val * 100:.2f}%)")

from PIL import Image
map_path = "C:/Users/Mahen/AppData/Local/Programs/Python/Python311/Lib/site-packages/andhra-pradesh-blank-map.jpg"
map_img = Image.open(map_path)

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(18, 18))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8])

'''now the PLOTTING'''

import cartopy.crs as ccrs


ax1 = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax1.imshow(map_img, extent=[andhra_lon_min, andhra_lon_max, andhra_lat_min, andhra_lat_max], alpha=0.7)
contour1 = ax1.contourf(ds_andhra.LONGITUDE.values[::2],ds_andhra.LATITUDE.values[::2],test_actual_reshaped.mean(axis=0),cmap='coolwarm', alpha=0.8, levels=20)
plt.colorbar(contour1, ax=ax1, label='Rainfall (mm)')
ax1.set_title(f"Test Set: Actual Rainfall (Spatial Avg) - Overall R²: {r2_test * 100:.2f}%")
ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax2.imshow(map_img, extent=[andhra_lon_min, andhra_lon_max, andhra_lat_min, andhra_lat_max], alpha=0.7)
contour2 = ax2.contourf(ds_andhra.LONGITUDE.values[::2],ds_andhra.LATITUDE.values[::2], test_predict_reshaped.mean(axis=0),cmap='coolwarm', alpha=0.8, levels=20)
plt.colorbar(contour2, ax=ax2, label='Rainfall (mm)')
ax2.set_title(f"Test Set: Predicted Rainfall (Spatial Avg) - Overall R²: {r2_test * 100:.2f}%")
ax3 = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax3.imshow(map_img, extent=[andhra_lon_min, andhra_lon_max, andhra_lat_min, andhra_lat_max], alpha=0.7)
contour3 = ax3.contourf(ds_andhra.LONGITUDE.values[::2],ds_andhra.LATITUDE.values[::2],val_actual_reshaped.mean(axis=0),cmap='coolwarm', alpha=0.8, levels=20)
plt.colorbar(contour3, ax=ax3, label='Rainfall (mm)')
ax3.set_title(f"Validation Set: Actual Rainfall (Spatial Avg) - Overall R²: {r2_val * 100:.2f}%")
ax4 = plt.subplot(gs[1, 1], projection=ccrs.PlateCarree())
ax4.imshow(map_img, extent=[andhra_lon_min, andhra_lon_max, andhra_lat_min, andhra_lat_max], alpha=0.7)
contour4 = ax4.contourf(ds_andhra.LONGITUDE.values[::2],ds_andhra.LATITUDE.values[::2],val_predict_reshaped.mean(axis=0),cmap='coolwarm', alpha=0.8, levels=20)
plt.colorbar(contour4, ax=ax4, label='Rainfall (mm)')
ax4.set_title(f"Validation Set: Predicted Rainfall (Spatial Avg) - Overall R²: {r2_val * 100:.2f}%")
plt.tight_layout()
plt.show()
