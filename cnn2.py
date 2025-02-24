import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.gridspec as gridspec

file_path = "D:/IMDrainfall_1901-2023.nc"

ds = xr.open_dataset(file_path, chunks={'TIME': 1000})
andhra_lon_min, andhra_lon_max = 76.0, 84.0
andhra_lat_min, andhra_lat_max = 12.0, 20.0
ds_andhra = ds.sel(LONGITUDE=slice(andhra_lon_min, andhra_lon_max), LATITUDE=slice(andhra_lat_min, andhra_lat_max))
train_data = ds_andhra.sel(TIME=slice("1901-01-01", "1990-12-31"))
val_data = ds_andhra.sel(TIME=slice("1991-01-01", "2011-12-31"))
test_data = ds_andhra.sel(TIME=slice("2012-01-01", "2023-12-31"))

rainfall_train = train_data["RAINFALL"].compute().values
rainfall_val = val_data["RAINFALL"].compute().values
rainfall_test = test_data["RAINFALL"].compute().values

# Interpolating missing values
rainfall_train = np.nan_to_num(rainfall_train, nan=np.nanmean(rainfall_train))
rainfall_val = np.nan_to_num(rainfall_val, nan=np.nanmean(rainfall_val))
rainfall_test = np.nan_to_num(rainfall_test, nan=np.nanmean(rainfall_test))

scaler = StandardScaler()
rainfall_train_scaled = scaler.fit_transform(rainfall_train.reshape(-1, 1)).reshape(rainfall_train.shape)
rainfall_val_scaled = scaler.transform(rainfall_val.reshape(-1, 1)).reshape(rainfall_val.shape)
rainfall_test_scaled = scaler.transform(rainfall_test.reshape(-1, 1)).reshape(rainfall_test.shape)

model = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(rainfall_train.shape[1], rainfall_train.shape[2], 1)),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(rainfall_train.shape[1] * rainfall_train.shape[2]),
    Reshape((rainfall_train.shape[1], rainfall_train.shape[2]))
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='mse')

# Early stopping and learning rate reduction
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(
    rainfall_train_scaled, rainfall_train_scaled,
    epochs=100,
    validation_data=(rainfall_val_scaled, rainfall_val_scaled),
    verbose=1,
    callbacks=[early_stop, lr_reduce]
)

# Plot training & validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

# Validation set predictions
val_predict = model.predict(rainfall_val_scaled)
val_actual_flat = rainfall_val.flatten()
val_predict_flat = val_predict.flatten()

plt.figure(figsize=(10, 5))
plt.scatter(val_actual_flat, val_predict_flat, alpha=0.5, label='Validation Predictions')
plt.xlabel('Actual Rainfall')
plt.ylabel('Predicted Rainfall')
plt.title('Validation Set: Actual vs. Predicted Rainfall')
plt.legend()
plt.grid()
plt.show()

test_predict = model.predict(rainfall_test_scaled)

test_actual_flat = rainfall_test.flatten()
test_predict_flat = test_predict.flatten()

mse_test = mean_squared_error(test_actual_flat, test_predict_flat)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(test_actual_flat, test_predict_flat)

print(f"Test Set - MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, R²: {r2_test:.2f} ({r2_test * 100:.2f}%)")

map_path = "C:/Users/Mahen/AppData/Local/Programs/Python/Python311/Lib/site-packages/andhra-pradesh-blank-map.jpg"
map_img = Image.open(map_path)
fig = plt.figure(figsize=(18, 18))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

ax1 = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax1.imshow(map_img, extent=[andhra_lon_min, andhra_lon_max, andhra_lat_min, andhra_lat_max], alpha=0.7)
contour1 = ax1.contourf(ds_andhra.LONGITUDE.values, ds_andhra.LATITUDE.values, rainfall_test.mean(axis=0), cmap='coolwarm', alpha=0.8, levels=20)
plt.colorbar(contour1, ax=ax1, label='Rainfall (mm)')
ax1.set_title(f"Test Set: Actual Rainfall (Spatial Avg) - Overall R²: {r2_test * 100:.2f}%")

ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax2.imshow(map_img, extent=[andhra_lon_min, andhra_lon_max, andhra_lat_min, andhra_lat_max], alpha=0.7)
contour2 = ax2.contourf(ds_andhra.LONGITUDE.values, ds_andhra.LATITUDE.values, test_predict.mean(axis=0), cmap='coolwarm', alpha=0.8, levels=20)
plt.colorbar(contour2, ax=ax2, label='Rainfall (mm)')
ax2.set_title(f"Test Set: Predicted Rainfall (Spatial Avg) - Overall R²: {r2_test * 100:.2f}%")

plt.tight_layout()
plt.show()
