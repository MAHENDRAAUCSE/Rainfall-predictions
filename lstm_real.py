import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec

# Load NetCDF data
file_path = "D:/IMDrainfall_1901-2023.nc"
ds = xr.open_dataset(file_path)

# Extract variables
time = ds['TIME'].values
rainfall = ds['RAINFALL'].values  # Assuming RAINFALL is the variable of interest
longitude = ds['LONGITUDE'].values
latitude = ds['LATITUDE'].values

# Interpolate NaN values in rainfall data
rainfall_interp = rainfall.copy()
rainfall_interp = xr.DataArray(rainfall_interp).interpolate_na(dim='TIME', method='linear')

# Flatten the rainfall data for scaling
rainfall_flat = rainfall_interp.values.flatten()

# Standardizing the data
scaler = StandardScaler()
rainfall_scaled = scaler.fit_transform(rainfall_flat.reshape(-1, 1))

# Reshaping back to the original dimensions after scaling
rainfall_scaled = rainfall_scaled.reshape(rainfall_interp.shape)

# Split data into train, validation, and test sets based on time
train_ds = ds.sel(TIME=slice("1901-01-01", "1990-12-31"))
val_ds = ds.sel(TIME=slice("1991-01-01", "2011-12-31"))
test_ds = ds.sel(TIME=slice("2012-01-01", "2023-12-31"))

# Extract rainfall data for each split
rainfall_train = train_ds['RAINFALL'].values
rainfall_val = val_ds['RAINFALL'].values
rainfall_test = test_ds['RAINFALL'].values

# Reshaping for sequences
def create_sequences(data, time_steps=90):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 0])  # Predicting rainfall
    return np.array(X), np.array(y)

time_steps = 90

# Create sequences for training, validation, and test sets
X_train, y_train = create_sequences(rainfall_train.flatten(), time_steps)
X_val, y_val = create_sequences(rainfall_val.flatten(), time_steps)
X_test, y_test = create_sequences(rainfall_test.flatten(), time_steps)

# Build the improved LSTM model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)),
    Bidirectional(LSTM(64, return_sequences=False, dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Predictions
y_pred = model.predict(X_test)
y_val_pred = model.predict(X_val)

# Calculate R^2 score
r2_test = r2_score(y_test, y_pred)
print(f"Test R^2 Score: {r2_test:.4f}")

# Visualization
plt.figure(figsize=(10, 5))
sns.histplot(y_test - y_pred.flatten(), bins=50, kde=True)
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

# Training loss curve
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.show()

# Actual vs Predicted values
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual', alpha=0.7)
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Rainfall")
plt.title("Actual vs Predicted Rainfall")
plt.legend()
plt.show()

# Scatter plot for actual vs predicted values
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred.flatten(), alpha=0.5)
plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("Actual vs Predicted Rainfall Scatter Plot")
plt.show()

# Additional Spatial Plots
fig = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8])

# Define geographic boundaries for the maps
andhra_lon_min, andhra_lon_max = longitude.min(), longitude.max()
andhra_lat_min, andhra_lat_max = latitude.min(), latitude.max()

def generate_spatial_plot(ax, data, title):
    """
    Generates a spatial plot using contourf to show rainfall predictions.
    ax: Matplotlib axis object
    data: 2D array of data to be plotted (e.g., rainfall)
    title: Title for the plot
    """
    ax.imshow(np.zeros((10, 10)), extent=[andhra_lon_min, andhra_lon_max, andhra_lat_min, andhra_lat_max], alpha=0.6)
    contour = ax.contourf(np.linspace(andhra_lon_min, andhra_lon_max, data.shape[1]),
                           np.linspace(andhra_lat_min, andhra_lat_max, data.shape[0]),
                           data.mean(axis=0), cmap='coolwarm', alpha=0.8)
    plt.colorbar(contour, ax=ax, label='Rainfall (mm)')
    ax.set_title(title)

# Actual vs Predicted Rainfall Spatial Plot for Test Set
ax1 = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
generate_spatial_plot(ax1, y_test.reshape((10, 10, -1)), "Test Set: Actual Rainfall (Avg)")

ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
generate_spatial_plot(ax2, y_pred.reshape((10, 10, -1)), "Test Set: Predicted Rainfall (Avg)")

# Actual vs Predicted Rainfall Spatial Plot for Validation Set
ax3 = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree())
generate_spatial_plot(ax3, y_val.reshape((10, 10, -1)), "Validation Set: Actual Rainfall (Avg)")

ax4 = plt.subplot(gs[1, 1], projection=ccrs.PlateCarree())
generate_spatial_plot(ax4, y_val_pred.reshape((10, 10, -1)), "Validation Set: Predicted Rainfall (Avg)")

plt.show()
