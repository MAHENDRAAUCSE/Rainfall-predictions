import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import xarray as xr
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

# Enable GPU memory growth to avoid crashes
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPUs detected: {physical_devices}")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPUs detected. Falling back to CPU.")

# File paths for seasonal NetCDF data
file_paths = {
    "monsoon": "D:/IMD_Rainfall/IMD_Rainfall/IMD_JJAS(Monsoon).nc",
    "post_monsoon": "D:/IMD_Rainfall/IMD_Rainfall/IMD_OND(postmonsoon).nc",
    "summer": "D:/IMD_Rainfall/IMD_Rainfall/IMD_MAM(premonsoon).nc",
    "winter": "D:/IMD_Rainfall/IMD_Rainfall/IMD_JF(winter).nc"
}

# Specify D: drive for memmap storage
memmap_dir = "D:/temp_memmap/"
os.makedirs(memmap_dir, exist_ok=True)

# Latitude and Longitude bounds for Andhra Pradesh
andhra_lon_min, andhra_lon_max = 76.0, 84.0
andhra_lat_min, andhra_lat_max = 12.0, 20.0


# Function to load and preprocess data
def load_and_preprocess_data(file_paths, time_steps=3):
    datasets = []
    date_ranges = []
    for season, path in file_paths.items():
        # Load the dataset
        ds = xr.open_dataset(path, chunks={'TIME': -1})
        rainfall = ds['RAINFALL'].astype(np.float32)
        # Subset the data for Andhra Pradesh region
        rainfall = rainfall.sel(LONGITUDE=slice(andhra_lon_min, andhra_lon_max),
                                LATITUDE=slice(andhra_lat_min, andhra_lat_max))
        # Interpolate NaN values along the TIME axis
        rainfall = rainfall.interpolate_na(dim='TIME', method='linear', fill_value="extrapolate")
        print(f"Loaded and interpolated {season} data with shape: {rainfall.shape}")
        # Append the processed data and dates
        datasets.append(rainfall.compute())  # Compute to load data into memory
        date_ranges.append(ds['TIME'].values)

    # Combine all datasets into one
    data = np.concatenate(datasets, axis=0)
    data = np.nan_to_num(data)

    # Normalize data using min-max scaling
    data_min = 1.0
    data_max = 6.0
    data = (data - data_min) / (data_max - data_min)

    dates = np.concatenate(date_ranges, axis=0)
    print(f"Combined dataset shape: {data.shape}")

    # Extract latitude and longitude values
    latitudes = rainfall['LATITUDE'].values
    longitudes = rainfall['LONGITUDE'].values

    # Create input (X) and target (y) arrays
    num_samples = len(data) - time_steps
    X_shape = (num_samples, data.shape[1], data.shape[2], time_steps)
    y_shape = (num_samples, data.shape[1], data.shape[2], 1)

    # Define file paths for memory-mapped arrays
    X_memmap_path = os.path.join(memmap_dir, r"X_memmap.dat")
    y_memmap_path = os.path.join(memmap_dir, r"y_memmap.dat")

    # Delete existing memmap files if they exist
    if os.path.exists(X_memmap_path):
        os.remove(X_memmap_path)
    if os.path.exists(y_memmap_path):
        os.remove(y_memmap_path)

    # Create memory-mapped arrays
    X = np.memmap(X_memmap_path, dtype=np.float32, mode='w+', shape=X_shape)
    y = np.memmap(y_memmap_path, dtype=np.float32, mode='w+', shape=y_shape)
    sample_dates = []
    for i in range(num_samples):
        X[i] = np.moveaxis(data[i:i + time_steps], 0, -1)
        y[i] = data[i + time_steps].reshape(data.shape[1], data.shape[2], 1)
        sample_dates.append(dates[i + time_steps])
        if i % 5000 == 0:
            print(f"Processed {i}/{num_samples} samples")
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    return X_memmap_path, y_memmap_path, np.array(sample_dates), latitudes, longitudes


# Load and preprocess data
X_memmap_path, y_memmap_path, sample_dates, latitudes, longitudes = load_and_preprocess_data(file_paths, time_steps=3)

# Free up memory
gc.collect()

# Create masks for train, test, and validation sets
train_mask = (sample_dates >= np.datetime64("1901-01-01")) & (sample_dates <= np.datetime64("1990-12-31"))
test_mask = (sample_dates >= np.datetime64("1991-01-01")) & (sample_dates <= np.datetime64("2011-12-31"))
val_mask = (sample_dates >= np.datetime64("2012-01-01")) & (sample_dates <= np.datetime64("2023-12-31"))

# Load data into memory-mapped arrays
X_train = np.memmap(X_memmap_path, dtype=np.float32, mode='r',
                    shape=(len(sample_dates[train_mask]), latitudes.size, longitudes.size, 3))
y_train = np.memmap(y_memmap_path, dtype=np.float32, mode='r',
                    shape=(len(sample_dates[train_mask]), latitudes.size, longitudes.size, 1))
X_test = np.memmap(X_memmap_path, dtype=np.float32, mode='r',
                   shape=(len(sample_dates[test_mask]), latitudes.size, longitudes.size, 3))
y_test = np.memmap(y_memmap_path, dtype=np.float32, mode='r',
                   shape=(len(sample_dates[test_mask]), latitudes.size, longitudes.size, 1))
X_val = np.memmap(X_memmap_path, dtype=np.float32, mode='r',
                  shape=(len(sample_dates[val_mask]), latitudes.size, longitudes.size, 3))
y_val = np.memmap(y_memmap_path, dtype=np.float32, mode='r',
                  shape=(len(sample_dates[val_mask]), latitudes.size, longitudes.size, 1))

print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test shape: X={X_test.shape}, y={y_test.shape}")
print(f"Validation shape: X={X_val.shape}, y={y_val.shape}")


# Use tf.data API for efficient data loading
def create_dataset(X_memmap_path, y_memmap_path, indices, batch_size):
    X = np.memmap(X_memmap_path, dtype=np.float32, mode='r', shape=(len(indices), latitudes.size, longitudes.size, 3))
    y = np.memmap(y_memmap_path, dtype=np.float32, mode='r', shape=(len(indices), latitudes.size, longitudes.size, 1))
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


batch_size = 8  # Adjusted batch size
train_indices = np.where(train_mask)[0]
val_indices = np.where(val_mask)[0]
test_indices = np.where(test_mask)[0]
train_dataset = create_dataset(X_memmap_path, y_memmap_path, train_indices, batch_size)
val_dataset = create_dataset(X_memmap_path, y_memmap_path, val_indices, batch_size)
test_dataset = create_dataset(X_memmap_path, y_memmap_path, test_indices, batch_size)

# Define and compile the model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(latitudes.size, longitudes.size, 3)),
    BatchNormalization(),
    Dropout(0.3),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(latitudes.size * longitudes.size, activation='linear'),
    Reshape((latitudes.size, longitudes.size, 1))
])
optimizer = Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
checkpoint = ModelCheckpoint('D:/best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=20,
                    callbacks=[reduce_lr, checkpoint])

# Evaluate the model
test_loss, test_mae = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Predictions
y_pred_test = model.predict(X_test)
y_pred_val = model.predict(X_val)


# Post-process predictions with Gaussian smoothing
def smooth_predictions(predictions, sigma=1):
    smoothed_predictions = []
    for pred in predictions:
        smoothed_pred = gaussian_filter(pred[:, :, 0], sigma=sigma)
        smoothed_predictions.append(smoothed_pred)
    return np.array(smoothed_predictions)


y_pred_test_smoothed = smooth_predictions(y_pred_test)
y_pred_val_smoothed = smooth_predictions(y_pred_val)

# Denormalize the data
data_min = 1.0
data_max = 6.0


def denormalize(data):
    return data * (data_max - data_min) + data_min


rainfall_test_denorm = denormalize(y_test[:, :, :, 0])
forecast_actual_denorm = denormalize(y_pred_test_smoothed)
rainfall_val_denorm = denormalize(y_val[:, :, :, 0])
val_forecast_actual_denorm = denormalize(y_pred_val_smoothed)

# Define fixed levels for contour plots
fixed_levels = np.linspace(1.0, 6.0, 20)

# Load custom map image for Andhra Pradesh
map_path = "C:/Users/Mahen/AppData/Local/Programs/Python/Python311/Lib/site-packages/andhra-pradesh-blank-map.jpg"
custom_map = Image.open(map_path).convert("RGB")
custom_map = np.array(custom_map, dtype=np.uint8)


# Define the plot_map function
def plot_map(ax, data, title, latitudes, longitudes):
    """
    Plots rainfall data on a map of Andhra Pradesh.

    Parameters:
        ax: Matplotlib Axes object
        data: Rainfall data (2D array or 3D array with time axis)
        title: Title for the subplot
        latitudes: Latitude values for the Andhra Pradesh region
        longitudes: Longitude values for the Andhra Pradesh region
    """
    # If data has a time axis, average over it
    if len(data.shape) == 3:
        data = data.mean(axis=0)

    # Create filled contour plot
    contour = ax.contourf(longitudes, latitudes, data,
                          cmap='coolwarm', alpha=0.8, levels=fixed_levels)

    # Add colorbar
    plt.colorbar(contour, ax=ax, label='Rainfall (mm)')

    # Set title
    ax.set_title(title)

    # Add custom map overlay
    img_extent = [andhra_lon_min, andhra_lon_max, andhra_lat_min, andhra_lat_max]
    ax.imshow(custom_map, extent=img_extent, transform=ccrs.PlateCarree(), alpha=0.5)


# Prepare datasets for plotting
rainfall_test = rainfall_test_denorm  # Actual rainfall from test set
forecast_actual = forecast_actual_denorm  # Smoothed predicted rainfall from test set
rainfall_val = rainfall_val_denorm  # Actual rainfall from validation set
val_forecast_actual = val_forecast_actual_denorm  # Smoothed predicted rainfall from validation set

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 18), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot actual and predicted rainfall for test and validation sets
plot_map(axes[0, 0], rainfall_test, "Test Set: Actual Rainfall", latitudes, longitudes)
plot_map(axes[0, 1], forecast_actual, "Test Set: Predicted Rainfall (Smoothed)", latitudes, longitudes)
plot_map(axes[1, 0], rainfall_val, "Validation Set: Actual Rainfall", latitudes, longitudes)
plot_map(axes[1, 1], val_forecast_actual, "Validation Set: Predicted Rainfall (Smoothed)", latitudes, longitudes)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()