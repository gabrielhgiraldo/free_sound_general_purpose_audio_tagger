import keras
from flask import Flask, render_template
from flask import request
import flask
app = Flask(__name__)
from keras.models import load_model
import librosa
import wave
import numpy as np
import pandas as pd
import pickle
def get_model():
    model = load_model('../best_model.h5')
    return model
def normalize_audio(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5
def window_data(data, input_length):
    data_length = len(data)
    num_windows = np.ceil(data_length/input_length)
    return np.array_split(data,num_windows)
def adjust_audio_length(data,input_length):
       # print("adjusted audio length")
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        return data
def transform_data(data,sampling_rate=44100,use_mel_spec=False,n_mels=128):
        if use_mel_spec:
            data = librosa.feature.melspectrogram(data,sr=sampling_rate,n_mels=n_mels)
        else:
            data = preprocessing_fn(data)[:, np.newaxis]
        return data

@app.route("/")
def get_page():
    with open("audio_tagger.html", 'r') as viz_file:
        return viz_file.read()

@app.route("/label_file", methods=['POST'])
def label_file():
    sampling_rate=44100
    audio_duration=2
    audio_length = sampling_rate * audio_duration
    audio_file = request.files['audio']
    model = get_model()
    name="audio_file.wav"
    #save audio file locally
    with wave.open(name,'wb') as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(44100)
        file.writeframesraw(audio_file.read())
    #process audio
    data, _ = librosa.core.load(name, sr=sampling_rate,
                                        res_type='kaiser_fast')
    #trim silence from data
    data, _ = librosa.effects.trim(data)
    #chop audio into pieces that are correct length for output
    #data = window_data(data,audio_length)
    data = [data]
    data = [adjust_audio_length(datum,audio_length) for datum in data]
    data = [transform_data(datum,sampling_rate) for datum in data]
    #predict tags
    X=np.array(data)
    predictions = model.predict(X)
    with open('../labels.pkl','rb+') as file:
        LABELS = pickle.load(file)
    label_ind = [np.argsort(-prediction_set)[:3] for prediction_set in predictions]
    pred_labels = [np.array(LABELS)[ind].tolist() for ind in label_ind]   
    probs = [np.take(predictions[i], ind).tolist() for i,ind in enumerate(label_ind)]
    return flask.jsonify(classes=pred_labels, probabilities = probs)

if __name__ == '__main__':
    app.run(debug=True, user_reloader=False)
