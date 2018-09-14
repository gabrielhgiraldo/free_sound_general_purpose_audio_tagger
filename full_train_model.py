from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution1D,MaxPooling1D,ZeroPadding1D
from keras.layers import GlobalMaxPooling1D
from keras import optimizers
import keras.utils.data_utils as d_utils
import keras.utils.np_utils as n_utils
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
import pandas as pd
import os
from sklearn.cross_validation import StratifiedKFold
import librosa
import shutil
import numpy as np
class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2, n_classes=41,
                 use_mel_spec=False,
                 n_mels=128,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001, 
                 max_epochs=50, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.use_mel_spec = use_mel_spec
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        elif self.use_mel_spec:
            self.dim = (n_mels, self.audio_duration*60)
        else:
            self.dim = (self.audio_length, 1)
# def normalize_audio(audio):
#     return audio/max(audio)
def normalize_audio(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5
class DataGenerator(d_utils.Sequence):
    def __init__(self, config, data_dir, list_IDs, labels=None, 
                 batch_size=64, preprocessing_fn=lambda x: x):
        self.config = config
        self.data_dir = data_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.preprocessing_fn = preprocessing_fn
        self.on_epoch_end()
        self.dim = self.config.dim

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        
    def transform_data(self,data):
        if self.config.use_mfcc:
            data = librosa.feature.mfcc(data, sr=self.config.sampling_rate,
                                               n_mfcc=self.config.n_mfcc)
            data = np.expand_dims(data, axis=-1)
        elif self.config.use_mel_spec:
            data = librosa.feature.melspectrogram(data,sr=self.config.sampling_rate,n_mels=self.config.n_mels)
        else:
            data = self.preprocessing_fn(data)[:, np.newaxis]
        return data
    
    def adjust_audio_length(self,data,input_length):
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
       # print(len(data))
        return data
#             if len(data) >= input_length:
#                 data = data[:input_length]
#             else:
#                 data = np.pad(data,input_length-len(data),"constant")
    def __data_generation(self, list_IDs_temp):
        cur_batch_size = len(list_IDs_temp)
        X = np.empty((cur_batch_size, *self.dim))

        input_length = self.config.audio_length
        for i, ID in enumerate(list_IDs_temp):
            file_path = self.data_dir + ID
            
            # Read and Resample the audio
            data, _ = librosa.core.load(file_path, sr=self.config.sampling_rate,
                                        res_type='kaiser_fast')
            
            #fixing lengths of files
            # Random offset / Padding
            data = self.adjust_audio_length(data,input_length)
            #other preprocessing
            data = self.transform_data(data)
         
            X[i,] = data

        if self.labels is not None:
            y = np.empty(cur_batch_size, dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.labels[ID]
            return X, n_utils.to_categorical(y, num_classes=self.config.n_classes)
        else:
            return X
def get_1d_dummy_model(config):
    
    nclass = config.n_classes
    input_length = config.audio_length
    
#     inp = Input(shape=(input_length,1))
    model = Sequential()
    model.add(Convolution1D(16, 9, activation='relu', padding="valid",input_shape=(input_length,1)))
    model.add(Convolution1D(16, 9, activation='relu', padding="valid"))
    #model.add(MaxPooling1D(16))
    #model.add(Dropout(rate=0.1))
    model.add(Dense(nclass, activation='softmax'))

#     model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model
def get_conv_model(config):
    
    nclass = config.n_classes
    input_length = config.audio_length
    
#     inp = Input(shape=(input_length,1))
    model = Sequential()
    model.add(Convolution1D(16, 9, activation='relu', padding="valid",input_shape=(input_length,1)))
    model.add(Convolution1D(16, 9, activation='relu', padding="valid"))
    model.add(MaxPooling1D(16))
    model.add(Dropout(rate=0.1))
    
    model.add(Convolution1D(32, 3, activation='relu', padding="valid"))
    model.add(Convolution1D(32, 3, activation='relu', padding="valid"))
    model.add(MaxPooling1D(4))
    model.add(Dropout(rate=0.1))
    
    model.add(Convolution1D(32, 3, activation='relu', padding="valid"))
    model.add(Convolution1D(32, 3, activation='relu', padding="valid"))
    model.add(MaxPooling1D(4))
    model.add(Dropout(rate=0.1))
    
    model.add(Convolution1D(256, 3, activation='relu', padding="valid"))
    model.add(Convolution1D(256, 3, activation='relu', padding="valid"))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(rate=0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(1028, activation='relu'))
    model.add(Dense(nclass, activation='softmax'))

    opt = optimizers.Adam(config.learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model
def train_test_model(train, config, quick_run=False):
    if quick_run:
        train = train.sample(2000)
        config = Config(sampling_rate=100, audio_duration=1, n_folds=2, max_epochs=1)

    PREDICTION_FOLDER = "predictions_1d_conv"
    if not os.path.exists(PREDICTION_FOLDER):
        os.mkdir(PREDICTION_FOLDER)
    if os.path.exists('logs/' + PREDICTION_FOLDER):
        shutil.rmtree('logs/' + PREDICTION_FOLDER)

    skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)

    for i, (train_split, val_split) in enumerate(skf):
        train_set = train.iloc[train_split]
        val_set = train.iloc[val_split]
        checkpoint = ModelCheckpoint(f'best_{i}.h5', monitor='val_loss', verbose=1, save_best_only=True)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%d'%i, write_graph=True)

        callbacks_list = [checkpoint, early, tb]
        #callbacks_list = [early]
        print("Fold: ", i)
        print("#"*50)
        if not quick_run:
            model = get_conv_model(config)
        else:
            model = get_1d_dummy_model(config)
        train_generator = DataGenerator(config, 'data/audio_train/', train_set.index, 
                                        train_set.label_idx, batch_size=64,
                                        preprocessing_fn=normalize_audio)
        val_generator = DataGenerator(config, 'data/audio_train/', val_set.index, 
                                      val_set.label_idx, batch_size=64,
                                      preprocessing_fn=normalize_audio)

        history = model.fit_generator(train_generator, callbacks=callbacks_list, validation_data=val_generator,
                                      epochs=config.max_epochs, use_multiprocessing=True, workers=6, max_queue_size=20)

        model.load_weights(f'best_{i}.h5')

        # Save train predictions
        train_generator = DataGenerator(config, 'data/audio_train/', train.index, batch_size=128,
                                        preprocessing_fn=normalize_audio)
        predictions = model.predict_generator(train_generator, use_multiprocessing=True, 
                                              workers=6, max_queue_size=20, verbose=1)
        np.save(PREDICTION_FOLDER + "/train_predictions_%d.npy"%i, predictions)

        # Save test predictions
        test_generator = DataGenerator(config, 'data/audio_test/', test.index, batch_size=128,
                                        preprocessing_fn=normalize_audio)
        predictions = model.predict_generator(test_generator, use_multiprocessing=True, 
                                              workers=6, max_queue_size=20, verbose=1)
        np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy"%i, predictions)

        #Make a submission file
        top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
        predicted_labels = [' '.join(list(x)) for x in top_3]
        test['label'] = predicted_labels
        test[['label']].to_csv(PREDICTION_FOLDER + "/predictions_%d.csv"%i)
        #model.save_weights(f'best_{i}.h5')
    model = get_conv_model(config)
    model.load_weights(f'best_{len(skf)-1}')
    model.save('best_model.h5')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/sample_submission.csv')
LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train.set_index("fname", inplace=True)
test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])

config = Config(sampling_rate=10000, audio_duration=1, n_folds=2, learning_rate=0.001,max_epochs=20)
train_test_model(train,config,quick_run=False)