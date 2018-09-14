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
def get_model():
    model = load_model('../best_model.h5')
    return model
def normalize_audio(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5
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
def transform_data(data,sampling_rate=10000,use_mel_spec=False,n_mels=128,use_mfcc=False,n_mfcc=10,preprocessing_fn=normalize_audio):
        if use_mfcc:
            data = librosa.feature.mfcc(data, sr=sampling_rate,
                                               n_mfcc=n_mfcc)
            data = np.expand_dims(data, axis=-1)
        elif use_mel_spec:
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
    audio_file = request.files['audio']
    model = get_model()
    #temporarily write image to disk
    name="audio_file.wav"
    with wave.open(name,'wb') as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(44100)
        file.writeframesraw(audio_file.read())
    #audio_file.save(name)
    #process audio
    data, sr = librosa.core.load(name, sr=10000,
                                        res_type='kaiser_fast')
    audio_length = sr * 1

    data = adjust_audio_length(data,audio_length)
    data = transform_data(data,sr)
    #predict tags
    X=np.empty((1,audio_length,1))
    X[0,] = data
    predictions = model.predict(X)
    train = pd.read_csv('../data/train.csv')
    LABELS = list(train.label.unique())
    top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
    return flask.jsonify(classes=top_3.tolist())

if __name__ == '__main__':
    app.run(debug=True, user_reloader=False)
