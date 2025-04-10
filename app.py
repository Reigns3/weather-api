from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
import numpy as np
import tensorflow as tf
import joblib
import xarray as xr
import logging
from datetime import datetime, timedelta
import os
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# GitHub raw URLs for files or Google Drive URL in case they don't exist
MODEL_URL = "https://drive.google.com/uc?id=1ftr8CysyjDUgUvsr2JmEV7K9TEiAQjKa"  # Use your Google Drive URL
TEST_DATA_URL = "https://raw.githubusercontent.com/Reigns3/weather-api/main/data/X_test_small.npy"
SCALER_URL = "https://raw.githubusercontent.com/Reigns3/weather-api/main/data/scaler.pkl"

MODEL_PATH = "models/convlstm_final_v4.h5"
TEST_DATA_PATH = "data/X_test_small.npy"
SCALER_PATH = "data/scaler.pkl"

def download_file(url, destination):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    if not os.path.exists(destination):
        logger.info(f"Downloading {url} to {destination}...")
        if "drive.google.com" in url:
            import gdown
            gdown.download(url, destination, quiet=False)
        else:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logger.info(f"Downloaded {destination}")
    else:
        logger.info(f"File {destination} already exists, skipping download.")

try:
    logger.info("Downloading required files...")
    download_file(MODEL_URL, MODEL_PATH)
    download_file(TEST_DATA_URL, TEST_DATA_PATH)
    download_file(SCALER_URL, SCALER_PATH)

    logger.info("Loading model...")
    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
    )

    logger.info("Loading scaler...")
    scaler = joblib.load(SCALER_PATH)

    logger.info("Loading dataset...")
    ds = xr.open_dataset('Dataset/final_weather_data.nc')

    logger.info("Loading test data...")
    X_test = np.load(TEST_DATA_PATH)
    logger.info(f"X_test shape: {X_test.shape}")

except Exception as e:
    logger.error(f"Failed to load resources: {e}")
    raise

variables = ['d2m', 't2m', 'sp', 'u10', 'v10', 'tcc', 'cp', 'lsp', 'tp']
temp_idx = variables.index('t2m')
precip_idx = variables.index('tp')
cloud_idx = variables.index('tcc')
wind_u_idx = variables.index('u10')
wind_v_idx = variables.index('v10')
dewpoint_idx = variables.index('d2m')

pred_start_date = datetime(2024, 12, 19, 19, 0)

@app.route('/weather', methods=['GET'])
def get_weather():
    try:
        logger.info("Processing weather request...")
        lat = float(request.args.get('lat', 48.5))
        lon = float(request.args.get('lon', 16.0))
        date_str = request.args.get('date')
        time_str = request.args.get('time', '00:00')

        if not (46.5 <= lat <= 49.0 and 9.5 <= lon <= 17.5):
            logger.warning("Coordinates out of bounds")
            return jsonify({'error': 'Latitude/Longitude out of bounds'}), 400

        lat_idx = np.abs(ds.latitude.values - lat).argmin()
        lon_idx = np.abs(ds.longitude.values - lon).argmin()
        logger.debug(f"Nearest grid point: lat_idx={lat_idx}, lon_idx={lon_idx}")

        seq_idx = -1
        if date_str:
            try:
                datetime_str = f"{date_str} {time_str}"
                req_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
                delta = req_datetime - pred_start_date
                seq_idx = int(delta.total_seconds() // 3600)
                if not (0 <= seq_idx < X_test.shape[0]):
                    return jsonify({'error': 'Date/time out of test range (2024-12-19 19:00 to 2025-01-01 00:00)'}), 400
            except ValueError:
                return jsonify({'error': 'Invalid date/time format, use YYYY-MM-DD HH:MM'}), 400

        logger.debug(f"Using sequence index: {seq_idx}")
        recent_input = X_test[seq_idx:seq_idx+1]
        logger.debug(f"Input shape: {recent_input.shape}")

        y_pred = model.predict(recent_input)
        logger.debug(f"Prediction shape: {y_pred.shape}")

        y_pred_flat = y_pred.reshape(-1, 9)
        y_pred_unscaled = scaler.inverse_transform(y_pred_flat).reshape(1, 11, 33, 9)

        y_pred_unscaled[:, :, :, temp_idx] -= 273.15
        y_pred_unscaled[:, :, :, dewpoint_idx] -= 273.15
        y_pred_unscaled[:, :, :, precip_idx] = np.maximum(y_pred_unscaled[:, :, :, precip_idx] * 1000, 0)  # Clip to 0
        y_pred_unscaled[:, :, :, 6] = np.maximum(y_pred_unscaled[:, :, :, 6] * 1000, 0)  # cp
        y_pred_unscaled[:, :, :, 7] = np.maximum(y_pred_unscaled[:, :, :, 7] * 1000, 0)  # lsp

        temp = y_pred_unscaled[0, lat_idx, lon_idx, temp_idx]
        precip = y_pred_unscaled[0, lat_idx, lon_idx, precip_idx]
        cloud = y_pred_unscaled[0, lat_idx, lon_idx, cloud_idx]
        wind_u = y_pred_unscaled[0, lat_idx, lon_idx, wind_u_idx]
        wind_v = y_pred_unscaled[0, lat_idx, lon_idx, wind_v_idx]
        dewpoint = y_pred_unscaled[0, lat_idx, lon_idx, dewpoint_idx]

        precip_chance = float(precip > 0.1)
        precip_type = 'Snow' if temp < 0 and precip > 0 else 'Rain' if precip > 0 else 'None'
        # Adjust condition based on time of day
        req_hour = req_datetime.hour if date_str else 0  # Default to 00:00 if no date
        is_daytime = 6 <= req_hour < 18  # Roughly sunrise to sunset
        if precip > 0:
            condition = 'Snowy' if temp < 0 else 'Rainy'
        else:
            condition = 'Sunny' if is_daytime and cloud < 0.3 else 'Clear' if cloud < 0.3 else 'Cloudy' if cloud < 0.7 else 'Overcast'

        rh = 100 * np.exp((17.625 * (dewpoint - temp)) / (243.04 + temp))
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        wind_dir = get_wind_direction(wind_u, wind_v)

        weather_data = {
            'temperature': float(temp),
            'precipitation': {
                'chance': precip_chance,
                'type': precip_type,
                'amount': float(precip),
                'timing': 'Now' if precip > 0 else 'None'
            },
            'condition': condition,
            'humidity': float(rh),
            'wind': {'speed': float(wind_speed), 'direction': wind_dir},
            'predicted_date': (pred_start_date + timedelta(hours=seq_idx)).strftime('%Y-%m-%d %H:%M')
        }
        logger.info("Request processed successfully")
        return jsonify(weather_data)

    except Exception as e:
        logger.error(f"Error in get_weather: {str(e)}", exc_info=True)
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500

def get_wind_direction(u, v):
    angle = np.degrees(np.arctan2(v, u)) % 360
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    idx = int((angle + 22.5) / 45) % 8
    return directions[idx]

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=5000)
