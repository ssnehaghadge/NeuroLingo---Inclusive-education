from flask import Flask, request, jsonify
import speech_recognition as sr
import os
import io
import pyaudio
import numpy as np

app = Flask(__name__)

# Initialize the recognizer
r = sr.Recognizer()

# Define the audio stream parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 # Mono audio is sufficient for speech recognition
RATE = 16000 # 16kHz is a good sample rate for speech

AMPLIFICATION_FACTOR = 1.5 

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    try:
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        # Manually process the audio data to amplify the signal
        audio_data_np = np.frombuffer(audio_data, dtype=np.int16)
        amplified_data_np = audio_data_np * AMPLIFICATION_FACTOR
        amplified_data = amplified_data_np.astype(np.int16).tobytes()

        # Create an AudioData object from the amplified audio bytes
        audio_to_transcribe = sr.AudioData(amplified_data, RATE, 2)
        
        text = r.recognize_google(audio_to_transcribe)
        
        # Save the transcribed text to a notes file on the server
        with open("notes.txt", "a") as f:
            f.write(text + "\n")

        return jsonify({"transcript": text})
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 500
    except sr.RequestError as e:
        return jsonify({"error": f"API request failed: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
