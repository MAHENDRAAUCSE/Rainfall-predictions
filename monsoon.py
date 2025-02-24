import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import cartopy.crs as ccrs

file_path = "D:/IMD_Rainfall/IMD_Rainfall/IMD_JJAS(Monsoon).nc"
ds = xr.open_dataset(file_path, chunks={'TIME': 1000})
andhra_lon_min, andhra_lon_max = 76.0, 84.0
andhra_lat_min, andhra_lat_max = 12.0, 20.0
ds_andhra = ds.sel(LONGITUDE=slice(andhra_lon_min, andhra_lon_max), LATITUDE=slice(andhra_lat_min, andhra_lat_max))
train_data = ds_andhra.sel(TIME=slice("1901-01-01", "1990-12-31"))
test_data = ds_andhra.sel(TIME=slice("1991-01-01", "2011-12-31"))
val_data= ds_andhra.sel(TIME=slice("2012-01-01", "2023-12-31"))

rainfall_train = np.nan_to_num(train_data["RAINFALL"].compute().values, nan=np.nanmean(train_data["RAINFALL"].values))
rainfall_val = np.nan_to_num(val_data["RAINFALL"].compute().values, nan=np.nanmean(val_data["RAINFALL"].values))
rainfall_test = np.nan_to_num(test_data["RAINFALL"].compute().values, nan=np.nanmean(test_data["RAINFALL"].values))
scaler = RobustScaler()
rainfall_train_scaled = scaler.fit_transform(rainfall_train.reshape(-1, 1)).reshape(rainfall_train.shape)
rainfall_val_scaled = scaler.transform(rainfall_val.reshape(-1, 1)).reshape(rainfall_val.shape)
rainfall_test_scaled = scaler.transform(rainfall_test.reshape(-1, 1)).reshape(rainfall_test.shape)

# CNN Model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(rainfall_train.shape[1], rainfall_train.shape[2], 1)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(rainfall_train.shape[1] * rainfall_train.shape[2]),
    Reshape((rainfall_train.shape[1], rainfall_train.shape[2]))
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    rainfall_train_scaled, rainfall_train_scaled, epochs=5,
    validation_data=(rainfall_val_scaled, rainfall_val_scaled), verbose=1,
    callbacks=[early_stop]
)

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()
forecast = model.predict(rainfall_test_scaled)
forecast_actual = scaler.inverse_transform(forecast.reshape(-1, 1)).reshape(forecast.shape)
val_forecast = model.predict(rainfall_val_scaled)
val_forecast_actual = scaler.inverse_transform(val_forecast.reshape(-1, 1)).reshape(val_forecast.shape)
mse_test = mean_squared_error(rainfall_test.flatten(), forecast_actual.flatten())
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(rainfall_test.flatten(), forecast_actual.flatten())
r2_test = r2_score(rainfall_test.flatten(), forecast_actual.flatten())
bias = np.mean(forecast_actual - rainfall_test)
mse_val = mean_squared_error(rainfall_val.flatten(), val_forecast_actual.flatten())
rmse_val = np.sqrt(mse_val)
mae_val = mean_absolute_error(rainfall_val.flatten(), val_forecast_actual.flatten())
r2_val = r2_score(rainfall_val.flatten(), val_forecast_actual.flatten())
bias_val = np.mean(val_forecast_actual - rainfall_val)

print(f"Validation Set Metrics:")
print(f"MSE: {mse_val:.2f}, RMSE: {rmse_val:.2f}, MAE: {mae_val:.2f}, Bias: {bias_val:.2f}, R²: {r2_val * 100:.2f}%")

print(f"Test Set Metrics:")
print(f"MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, MAE: {mae_test:.2f}, Bias: {bias:.2f}, R²: {r2_test * 100:.2f}%")


fixed_levels = np.linspace(1.0, 6.0, 20)
map_path = "C:/Users/Mahen/AppData/Local/Programs/Python/Python311/Lib/site-packages/andhra-pradesh-blank-map.jpg"
custom_map = Image.open(map_path).convert("RGB")
custom_map = np.array(custom_map, dtype=np.uint8)
fig, axes = plt.subplots(2, 2, figsize=(18, 18), subplot_kw={'projection': ccrs.PlateCarree()})

def plot_map(ax, data, title):
    contour = ax.contourf(ds_andhra.LONGITUDE.values, ds_andhra.LATITUDE.values, data.mean(axis=0), cmap='coolwarm',
                          alpha=0.8, levels=fixed_levels)
    plt.colorbar(contour, ax=ax, label='Rainfall (mm)')
    ax.set_title(title)
    img_extent = [andhra_lon_min, andhra_lon_max, andhra_lat_min, andhra_lat_max]
    ax.imshow(custom_map, extent=img_extent, transform=ccrs.PlateCarree(), alpha=0.5)

plot_map(axes[0, 0], rainfall_test, "Test Set: Actual Rainfall")
plot_map(axes[0, 1], forecast_actual, "Test Set: Predicted Rainfall")
plot_map(axes[1, 0], rainfall_val, "Validation Set: Actual Rainfall")
plot_map(axes[1, 1], val_forecast_actual, "Validation Set: Predicted Rainfall")

plt.tight_layout()
plt.show()
