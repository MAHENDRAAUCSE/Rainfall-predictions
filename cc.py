import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K
import matplotlib.image as mpimg

# Load dataset
file_path = "D:/IMDrainfall_1901-2023.nc"
ds = xr.open_dataset(file_path)

# Define Andhra Pradesh's lat/lon range (approximate)
AP_LAT_MIN, AP_LAT_MAX = 12.6, 19.1
AP_LON_MIN, AP_LON_MAX = 76.7, 84.7

# Select only Andhra Pradesh region
ds_ap = ds.sel(LATITUDE=slice(AP_LAT_MIN, AP_LAT_MAX), LONGITUDE=slice(AP_LON_MIN, AP_LON_MAX))

# Extract rainfall data
rainfall = ds_ap['RAINFALL'].values  # Shape: (TIME, LATITUDE, LONGITUDE)

# Normalize rainfall values (0-1 scale)
scaler = MinMaxScaler()
rainfall = scaler.fit_transform(rainfall.reshape(-1, 1)).reshape(rainfall.shape)

# Define time splits
train_years = ds_ap.TIME.dt.year <= 1990
test_years = (ds_ap.TIME.dt.year > 1990) & (ds_ap.TIME.dt.year <= 2005)
val_years = ds_ap.TIME.dt.year > 2005

# Get datasets
X_train = rainfall[train_years]
X_test = rainfall[test_years]
X_val = rainfall[val_years]

def replace_nan_with_mean(data):
    nan_mask = np.isnan(data)
    mean_value = np.nanmean(data)  # Compute mean ignoring NaNs
    data[nan_mask] = mean_value
    return data

# Apply to all datasets
X_train = replace_nan_with_mean(X_train)
X_val = replace_nan_with_mean(X_val)
X_test = replace_nan_with_mean(X_test)

# Reshape for CNN (Adding channel dimension)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)

# Shift the rainfall data so that X contains past rainfall and y contains future rainfall
X_train, y_train = X_train[:-1], X_train[1:]
X_test, y_test = X_test[:-1], X_test[1:]
X_val, y_val = X_val[:-1], X_val[1:]

print("Updated Training Data Shape:", X_train.shape, y_train.shape)
print("Updated Testing Data Shape:", X_test.shape, y_test.shape)
print("Updated Validation Data Shape:", X_val.shape, y_val.shape)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(X_train.shape[1] * X_train.shape[2], activation='linear'),  # Output layer with correct dimensions
    Reshape((X_train.shape[1], X_train.shape[2], 1))  # Reshape back to original input shape
])

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
history = model.fit(
    X_train, y_train,
    epochs=5,  # Adjust based on performance
    batch_size=32,  # Adjust based on dataset size
    validation_data=(X_val, y_val),
    verbose=1  # Display progress
)

# Predict rainfall
predicted_rainfall = model.predict(X_test)
predicted_rainfall = predicted_rainfall.squeeze()  # Remove extra dimension
actual_rainfall = y_test.squeeze()

# Save model
model.save("D:/rainfall_cnn_model.keras")

# Load the Andhra Pradesh map image
ap_map = mpimg.imread("C:/Users/Mahen/AppData/Local/Programs/Python/Python311/Lib/site-packages/andhra-pradesh-blank-map.jpg")

# Plot the map as a background
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot Actual
axes[0].imshow(ap_map, extent=[AP_LON_MIN, AP_LON_MAX, AP_LAT_MIN, AP_LAT_MAX])
im1 = axes[0].imshow(actual_rainfall.mean(axis=0), cmap='Blues', origin='lower', alpha=0.6)
axes[0].set_title("Actual Rainfall")
plt.colorbar(im1, ax=axes[0])

# Plot Predicted
axes[1].imshow(ap_map, extent=[AP_LON_MIN, AP_LON_MAX, AP_LAT_MIN, AP_LAT_MAX])
im2 = axes[1].imshow(predicted_rainfall.mean(axis=0), cmap='Reds', origin='lower', alpha=0.6)
axes[1].set_title("Predicted Rainfall")
plt.colorbar(im2, ax=axes[1])

plt.show()
