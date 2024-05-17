from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# Load the dataset and preprocess it (this would be more complex in reality)
df = pd.read_csv(r"C:\Users\maria\Desktop\my_project\new.csv")
scaler = MinMaxScaler(feature_range=(0, 1))
df['Adj Close'] = scaler.fit_transform(df[['Adj Close']])

@app.route('/')
def index():
    return render_template('dm.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_type = data['model']
    features = np.array(data['features']).reshape(1, -1)

    if model_type == 'ARIMA':
        order = (5, 1, 0)
        model = ARIMA(df['Adj Close'], order=order)
        model_fit = model.fit()
        prediction = model_fit.forecast(steps=1)
    elif model_type == 'ANN':
        # Load your ANN model (Ensure the model is saved and loadable)
        model = load_model('ann_model.h5')
        prediction = model.predict(features)
    elif model_type == 'SARIMA':
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)
        model = SARIMAX(df['Adj Close'], order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        prediction = model_fit.forecast(steps=1)
    elif model_type == 'EXPONENTIAL':
        model = ExponentialSmoothing(df['Adj Close'], trend='add', seasonal='add', seasonal_periods=12)
        model_fit = model.fit()
        prediction = model_fit.forecast(steps=1)
    elif model_type == 'SVR':
        # Ensure you load a pre-trained SVR model
        model = load('svr_model.joblib')
        prediction = model.predict(features)

    response = {
        'prediction': prediction[0]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
