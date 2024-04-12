from flask import Flask, jsonify
from flask_cors import CORS
import pickle
import librosa
import numpy as np
import pyaudio
import wave

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
@app.route('/predict')
def predict():    
    # Constants for audio recording
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = "output.wav"

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open stream for recording
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    # Record audio data
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording complete.")

    # Stop recording and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save recorded audio to a WAV file
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print("Audio saved to:", WAVE_OUTPUT_FILENAME)
    # Process the audio file from the URL
    try:
        audio, sample_rate = librosa.load(WAVE_OUTPUT_FILENAME, res_type='kaiser_fast')
        
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
