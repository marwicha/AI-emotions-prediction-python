import os, sys
import base64
import glob
import json
import requests
import time

import librosa
from flask import Flask, flash, request, redirect, url_for, jsonify, abort
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import logging


from model import VGG
from transform import MelTranform, EmotionEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('HELLO WORLD')

# create the Flask application
app = Flask(__name__)
CORS(app)

save_path = os.path.join(sys.path[0], 'vgg_11_final_v1.pth')
model = None
mel_transform = None
emotion_encoder = None

HOST, PORT = 'localhost', 5000


ROUTE = 'speech_recognition/predict'
# get the url as a string
url = f'http://{HOST}:{PORT}/{ROUTE}'


@app.before_first_request
def initialize():
    global save_path
    global model
    global mel_transform
    global emotion_encoder
    print('Initialize model - MEL - Emotion')
    model = VGG('vgg11', True, save_path)
    model.eval()
    mel_transform = MelTranform()
    emotion_encoder = EmotionEncoder()

# create a method to send the data to the API when requested
@app.route("/speech_recognition/predict", methods=['POST'])
def send_data():
    """
    API route function to predict emotions from a waveform

    Call the API with URL '{HOST}:{PORT}/speech_recognition/predict/' posting:
        - "waveform": list of floats
        - "sample_rate": integer

    :return: JSON prediction probabilities for each emotion
    """
    try:
        # recover waveform and sample rate from POST
        data = request.get_json(force=True)

        if 'base64_waveform' in data:
            decode_string = base64.b64decode(data['base64_waveform'])
            wav_file = open("_temp.wav", "wb")
            wav_file.write(decode_string)
            data['waveform'], data['sample_rate'] = librosa.load("_temp.wav", sr=16000)

        waveform = data['waveform']

        sample_rate = data['sample_rate']

        # convert the waveform into a spectrogram
        spec = mel_transform(waveform, sample_rate)
        # predict the emotion from the spectrogram with the model
        pred, _ = model.predict_sample(spec)

        # convert the emotion probabiliies array into a comprehensive
        # JSON dictionnary with "emotion_name": probabily
        predictions = {
            emotion: pred[idx] * 1. # to float to be jsonified
            for idx, emotion in emotion_encoder.id_to_str.items()
        }
        # sort the dictionnary
        predictions = dict(sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        # convert into JSON format and send
        return jsonify(predictions)
    except Exception as e:
        print(e, 'error')
        abort(400)

# create a method to send the data to the API when requested
@app.route("/", methods=['GET'])
def home():
    """
    API route function to predict emotions from a waveform

    Call the API with URL '{HOST}:{PORT}/speech_recognition/predict/' posting:
        - "waveform": list of floats
        - "sample_rate": integer

    :return: JSON prediction probabilities for each emotion
    """
    try:
        print('home')
        return jsonify(['hello world'])
    except Exception as e:
        print(e, 'error')
        abort(400)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response


if __name__ == '__main__':
    app.run(debug=True)
