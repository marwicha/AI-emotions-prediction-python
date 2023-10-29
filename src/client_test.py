"""
    THIS IS A EXAMPLE OF HOW TO DAL WITH THE SPEECH EMOTION RECOGNITION API
"""

import os
import glob
import json
import requests
import time


import librosa

HOST, PORT = 'localhost', 5000


ROUTE = 'speech_recognition/predict'
# get the url as a string
url = f'http://{HOST}:{PORT}/{ROUTE}'


def call_speech_recognition_api(url, waveform, sample_rate):
    """
    Speech Recognition API request function

    :param url: url of the API to POST
    :param waveform: waveform to predict
    :param sample_rate: sample rate of the waveform
    :return: sorted JSON prediction probabilities for each emotion
    """
    # post data to the API
    data = {
        'waveform': list(waveform.astype('float64')),
        'sample_rate': sample_rate
    }
    # r = requests.post(url, json={'waveform':json.dumps([0]), 'sample_rate': SAMPLE_RATE})
    r = requests.post(url, json=data)

    # get the results if status code OK
    predictions = r.json() if r.status_code == 200 else {'Error status code': r.status_code}
    # sort the emotions given their probabilities
    predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
    return predictions


if __name__ == '__main__':
    # ignore warnings not to be spammed by librosa
    # sample rate to get read audio files (can be None)
    SAMPLE_RATE = 16000

    # set a list of audio files to read and predict
    audio_files_path = glob.glob(os.path.join('.', 'data_test', '*.*'))

    # read all the files as waveforms
    waveform_sample_rate_list = []
    for audio_path in glob.glob(os.path.join('.', 'data_test', '*.*')):
        try:
            waveform_sample_rate_list.append(librosa.load(audio_path, SAMPLE_RATE))
        except:
            print('Error reading', audio_path)
            import numpy as np
            waveform_sample_rate_list.append((np.zeros(32), SAMPLE_RATE))

    # prediction of emotion from each speech audio files
    for audio_path, (waveform, sample_rate) in zip(audio_files_path, waveform_sample_rate_list):
        print(f'Sending {len(waveform)} samples ({len(waveform)/SAMPLE_RATE:.2f}s)')
        # API POSTING
        t0 = time.time()
        predictions = call_speech_recognition_api(url, waveform, sample_rate)
        t1 = time.time()

        # only to produce a readable result on console.
        infos = ['speaker_id', 'emotion', 'level', 'sentence_id']
        infos_values = audio_path.split(os.sep)[-1].split('.')[0].split('_')
        d_infos = {key:value for key, value in zip(infos, infos_values)}
        d_infos['predictions'] = predictions
        d_infos['elapsed time'] = t1 - t0
        print(json.dumps(d_infos, indent=4))


    #
