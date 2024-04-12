from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import librosa
import numpy as np
import requests
from io import BytesIO

# Load the machine learning model
model = pickle.load(open("RF_model.pkl", 'rb'))

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# Define home route
@app.route('/')
def home():
    return 'Welcome to the Asphyxia Detection API'

# Define route to handle audio file URL and return prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains a URL parameter
    if 'url' not in request.form:
        return jsonify({'error': 'No URL parameter found'}), 400
    
    # Extract the URL parameter from the request
    audio_url = request.form['url']
    print('audio_url:', audio_url)
    # Process the audio file from the URL
    try:
        # Download the audio file from the URL
        response = requests.get(audio_url)
        
        # Check if the download was successful
        if response.status_code != 200:
            return jsonify({'error': 'Failed to download audio file from URL'}), 500
        
        # Load the audio file from the downloaded content
        audio_content = BytesIO(response.content)
        audio, sample_rate = librosa.load(audio_content, sr=None)
        
        # Extract features from audio
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

        # Make prediction using the loaded model
        prediction = model.predict(mfccs_scaled_features)[0]

        # Convert prediction to a standard Python integer
        prediction = int(prediction)

        # Prepare response based on prediction
        result = 'The patient has no Asphyxia' if prediction == 0 else 'The patient has Asphyxia'
        print('Result:', result)

        return jsonify({'result': result, 'prediction': prediction}), 200
    
    except Exception as e:
        print('Error processing', str(e))
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
