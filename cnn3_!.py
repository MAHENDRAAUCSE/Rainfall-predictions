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

# File paths for seasonal NetCDF data
season_files = {
    "monsoon": "D:/IMD_Rainfall/IMD_Rainfall/IMD_JJAS(Monsoon).nc",
    "post_monsoon": "D:/IMD_Rainfall/IMD_Rainfall/IMD_OND(postmonsoon).nc",
    "summer": "D:/IMD_Rainfall/IMD_Rainfall/IMD_MAM(premonsoon).nc",
    "winter": "D:/IMD_Rainfall/IMD_Rainfall/IMD_JF(winter).nc"
}

# Function to replace NaN values with seasonal mean
def replace_nan_with_mean(ds, variable='RAINFALL'):
    mean_value = ds[variable].mean(dim="TIME", skipna=True)  # Compute seasonal mean
    ds[variable] = ds[variable].fillna(mean_value).astype(np.float32)  # Convert to float32
    return ds


# Load datasets and clean NaN values
season_datasets = []
for season, file in season_files.items():
    ds = xr.open_dataset(file)  # Load dataset
    ds = replace_nan_with_mean(ds)  # Handle NaNs
    season_datasets.append(ds)

merged_dataset = xr.concat(season_datasets, dim="TIME").astype(np.float32)


# Select Andhra Pradesh region
andhra_lon_min, andhra_lon_max = 76.0, 84.0
andhra_lat_min, andhra_lat_max = 12.0, 20.0
ds_andhra = merged_dataset.sel(LONGITUDE=slice(andhra_lon_min, andhra_lon_max), LATITUDE=slice(andhra_lat_min, andhra_lat_max))

# Extract rainfall values and reshape
rainfall = ds_andhra['RAINFALL'].values  # Shape: (TIME, LAT, LON)

# Normalize rainfall
scaler = RobustScaler()
rainfall_scaled = scaler.fit_transform(rainfall.reshape(-1, 1)).reshape(rainfall.shape)

ds = xr.open_dataset(file, chunks={'TIME': 100})


# Define training, testing, and validation splits
train_years = ds_andhra.sel(TIME=slice("1900", "1990"))
test_years = ds_andhra.sel(TIME=slice("1991", "2005"))
val_years = ds_andhra.sel(TIME=slice("2006", "2023"))

# Extract and replace NaNs with the respective dataset mean
rainfall_train = np.nan_to_num(train_years["RAINFALL"].values, nan=np.nanmean(train_years["RAINFALL"].values))
rainfall_val = np.nan_to_num(val_years["RAINFALL"].values, nan=np.nanmean(val_years["RAINFALL"].values))
rainfall_test = np.nan_to_num(test_years["RAINFALL"].values, nan=np.nanmean(test_years["RAINFALL"].values))

# Normalize again after replacing NaNs
rainfall_train_scaled = scaler.fit_transform(rainfall_train.reshape(-1, 1)).reshape(rainfall_train.shape)
rainfall_val_scaled = scaler.transform(rainfall_val.reshape(-1, 1)).reshape(rainfall_val.shape)
rainfall_test_scaled = scaler.transform(rainfall_test.reshape(-1, 1)).reshape(rainfall_test.shape)

# Ensure the correct shape for CNN
rainfall_train_scaled = rainfall_train_scaled[..., np.newaxis]
rainfall_val_scaled = rainfall_val_scaled[..., np.newaxis]
rainfall_test_scaled = rainfall_test_scaled[..., np.newaxis]

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=rainfall_train_scaled.shape[1:]),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(rainfall_train_scaled.shape[1] * rainfall_train_scaled.shape[2]),
    Reshape((rainfall_train_scaled.shape[1], rainfall_train_scaled.shape[2]))
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(
    rainfall_train_scaled, rainfall_train_scaled, epochs=50,
    validation_data=(rainfall_val_scaled, rainfall_val_scaled), verbose=1,
    callbacks=[early_stop]
)

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

# Forecasting
forecast = model.predict(rainfall_test_scaled)
forecast_actual = scaler.inverse_transform(forecast.reshape(-1, 1)).reshape(forecast.shape)
val_forecast = model.predict(rainfall_val_scaled)
val_forecast_actual = scaler.inverse_transform(val_forecast.reshape(-1, 1)).reshape(val_forecast.shape)

# Compute metrics
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

print(f"Test Set Metrics:\nMSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, MAE: {mae_test:.2f}, Bias: {bias:.2f}, R²: {r2_test * 100:.2f}%")
print(f"Validation Set Metrics:\nMSE: {mse_val:.2f}, RMSE: {rmse_val:.2f}, MAE: {mae_val:.2f}, Bias: {bias_val:.2f}, R²: {r2_val * 100:.2f}%")


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
