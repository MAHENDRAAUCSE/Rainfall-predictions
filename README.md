# Rainfall Prediction using LSTM and CNN

## Description

This project implements machine learning models, specifically **Long Short-Term Memory (LSTM) networks** and **Convolutional Neural Networks (CNNs)**, to predict rainfall patterns based on historical **NetCDF** datasets. The models analyze rainfall trends and generate future predictions, visualizing the results through spatial heatmaps.

## Features

- **Data Preprocessing**: Handling missing values, standardization, and normalization of rainfall data.
- **LSTM Model**: Time-series forecasting using past rainfall data to predict future values.
- **CNN Model**: Spatial analysis using convolutional layers to predict rainfall distribution.
- **Training & Evaluation**: Uses **Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score** to assess model performance.
- **Visualization**: Generates error distribution plots, actual vs predicted comparisons, and spatial heatmaps.
- **AWS/GPU Optimization**: Code supports execution on **TensorFlow GPU** for improved efficiency.

## Technologies Used

- **Python 3.11.2**
- **TensorFlow & Keras** (LSTM, CNN models)
- **Xarray** (NetCDF file handling)
- **Matplotlib & Seaborn** (Visualization)
- **Cartopy** (Geospatial mapping)
- **Scikit-learn** (Data preprocessing & model evaluation)

## Dataset

The model utilizes **NetCDF files** containing rainfall data from **1901 to 2023**. The dataset is split into:

- **Training (1901-1990)**
- **Testing(1991-2011)**
- **Validation (2012-2023)**

## Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/RainfallPrediction.git
   cd RainfallPrediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the LSTM model:
   ```bash
   python lstm_real.py
   ```
4. Run the CNN model:
   ```bash
   python cnn_3.py
   ```

## Model Performance

The models were evaluated using standard metrics:

- **LSTM Model**: Achieved an R² score of 70.63\*\*%\*\* with RMSE of 2.55 mm.
- **CNN Model**: Produced spatially accurate predictions with 80.59\*\*% R² score\*\*.

## Future Improvements

- Improve model accuracy by experimenting with **hybrid LSTM-CNN architectures**.
- Optimize data processing for faster training.
- Deploy as a web-based rainfall prediction tool.

## Contributors

- **Mahendra (Lead Developer)**
- **Rasagna**

## License

This project is licensed under the **MIT License**.

