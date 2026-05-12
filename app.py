from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)

# Load merged_arima (replace with your actual data loading)
# Example: Load from CSV if saved
try:
    merged_arima = pd.read_csv('clustered_data_deduplicated.csv', parse_dates=['timestamp'])
except FileNotFoundError:
    print("Error: merged_arima.csv not found. Please save merged_arima from the notebook.")
    exit(1)

# Define features used in models
features = ['temperature', 'humidity', 'windSpeed', 'hour', 'Cluster', 
            'season_fall', 'season_spring', 'season_summer', 'season_winter']

# Load saved models
try:
    poly = joblib.load('models/poly_features.joblib')
    lr_model = joblib.load('models/lr_model.joblib')
    poly_model = joblib.load('models/poly_model.joblib')
    rf_model = joblib.load('models/rf_model.joblib')
    xgb_model = joblib.load('models/xgb_model.joblib')
    lstm_model = load_model('models/lstm_model.keras')
    with open('models/arima_model.pkl', 'rb') as f:
        arima_model = pickle.load(f)
    stacking_model = joblib.load('models/stacking_model.joblib')
except FileNotFoundError as e:
    print(f"Error: Model file not found: {e}")
    exit(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_name = data['model']
    start_date = pd.to_datetime(data['start_date'])
    end_date = pd.to_datetime(data['end_date'])

    # Filter data for the date range
    mask = (merged_arima['timestamp'] >= start_date) & (merged_arima['timestamp'] <= end_date)
    df_subset = merged_arima[mask]
    if df_subset.empty:
        return jsonify({'error': 'No data available for the selected date range'}), 400

    X_subset = df_subset[features].values
    y_actual = df_subset['demand'].values
    timestamps = df_subset['timestamp'].astype(str).tolist()

    # Generate predictions
    try:
        if model_name == 'ARIMA':
            start_idx = df_subset.index[0]
            end_idx = df_subset.index[-1]
            y_pred = arima_model.predict(start=start_idx, end=end_idx, dynamic=False).values
        elif model_name == 'Linear':
            y_pred = lr_model.predict(X_subset)
        elif model_name == 'Polynomial':
            X_poly_subset = poly.transform(X_subset)
            y_pred = poly_model.predict(X_poly_subset)
        elif model_name == 'Random Forest':
            y_pred = rf_model.predict(X_subset)
        elif model_name == 'XGBoost':
            y_pred = xgb_model.predict(X_subset)
        elif model_name == 'LSTM':
            X_lstm_subset = X_subset.reshape((X_subset.shape[0], 1, X_subset.shape[1]))
            y_pred = lstm_model.predict(X_lstm_subset, verbose=0).flatten()
        elif model_name == 'Stacking':
            y_pred = stacking_model.predict(X_subset)
        else:
            return jsonify({'error': 'Invalid model selected'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # Calculate metrics
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mask_nonzero = y_actual != 0
    mape = np.mean(np.abs((y_actual[mask_nonzero] - y_pred[mask_nonzero]) / y_actual[mask_nonzero])) * 100 if mask_nonzero.any() else float('inf')

    # Prepare response
    response = {
        'timestamps': timestamps,
        'actual': y_actual.tolist(),
        'predicted': y_pred.tolist(),
        'metrics': {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
