import flask
import joblib
import numpy as np
import requests
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Download and load the XGBoost model
MODEL_URL = "https://drive.google.com/uc?export=download&id=18KYrb2HassCBo0X3xx47I5PkU9A_3blq"
MODEL_PATH = "xgboost_model.pkl"

def download_model(url, path):
    try:
        print(f"Attempting to download model from {url}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully")
        else:
            raise Exception(f"Failed to download model: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

if not os.path.exists(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH)

try:
    model = joblib.load(MODEL_PATH)
    print("XGBoost model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.route('/')
def index():
    return flask.send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received prediction request:", request.get_json())
        data = request.get_json()
        if not data:
            raise Exception("No JSON data provided")

        host_pop = float(data['host_popularity'])
        guest_pop = float(data['guest_popularity'])
        genre = data['genre']
        pub_day = data['publication_day']
        ep_length = float(data['episode_length'])

        # Define categories for one-hot encoding (update based on your model)
        genres = ['Comedy', 'News', 'Education', 'True Crime']
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Preprocess inputs
        genre_encoded = [1 if g == genre else 0 for g in genres]
        day_encoded = [1 if d == pub_day else 0 for d in days]
        features = [host_pop, guest_pop, ep_length] + genre_encoded + day_encoded

        # Convert to numpy array for prediction
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        print(f"Prediction: {prediction}")

        return jsonify({'prediction': round(float(prediction), 2)})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
