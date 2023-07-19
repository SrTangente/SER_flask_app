import pandas as pd
import numpy as np
import pickle

import os
import sys
import librosa # for processing audio input
import librosa.display
import librosa.effects

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def pitch_shift(data, sr):
    """Apply pitch shifting to an audio signal."""
    n_steps = int(np.random.uniform(low=-5, high = 5))
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)

def mix_with_background_noise(data, noise_factor=0.025):
    """Mix an audio signal with background noise"""
    noise_amp = noise_factor*np.random.uniform()*np.amax(data)
    background_noise = noise_amp*np.random.normal(size=data.shape[0])
    return data + background_noise


def zcr(data, frame_length=2048, hop_length=512):
    '''Zero crossing rate measures the rate at which the signal changes its sign'''
    return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length))

def rmse(data, frame_length=2048, hop_length=512):
    '''Root Mean Square Energy of an audio signal. RMSE is a measure of the energy in the audio signal'''
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    '''Extract Mel-frequency Cepstral Coefficients (MFCCs) from an audio signal
    MFCCs are widely used features for representing the spectral characteristics of an audio signal'''
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr)
    return np.ravel(mfcc_feature.T) if flatten else mfcc_feature.T.flatten()


def extract_features(data, sr, frame_length=2048, hop_length=512):
    features = np.array([])
    features = np.hstack((features,
                          zcr(data, frame_length, hop_length),
                          rmse(data, frame_length, hop_length),
                          mfcc(data, sr, frame_length, hop_length)
                          ))
    return features

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files
    audio_data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    features1 = extract_features(audio_data, sample_rate)
    result = np.array(features1)

    # data with noise
    noise_data = mix_with_background_noise(audio_data)
    features2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, features2))  # stacking vertically

    # data with pitch shift
    pitch_shifted_data = pitch_shift(audio_data, sample_rate)
    features3 = extract_features(pitch_shifted_data, sample_rate)
    result = np.vstack((result, features3))  # stacking vertically

    # data with pitch shift and noise
    pitch_shifted_data_noise = mix_with_background_noise(pitch_shifted_data)
    features4 = extract_features(pitch_shifted_data_noise, sample_rate)
    result = np.vstack((result, features4))  # stacking vertically

    return result
    
def prepare_data():
    data_path = "./data/wav/"

    file_list = os.listdir(data_path)

    file_emotion = []
    file_path = []

    for file in file_list:
        file_path.append(data_path + file)
        
        em = file[5]
        if em=='W':
            file_emotion.append('angry')
        elif em=='E':
            file_emotion.append('disgust')
        elif em=='A':
            file_emotion.append('fear')
        elif em=='F':
            file_emotion.append('happy')
        elif em=='N':
            file_emotion.append('neutral')
        elif em=='T':
            file_emotion.append('sad')
        elif em=='L':
            file_emotion.append('boredom')

            
    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['path'])
    df = pd.concat([emotion_df, path_df], axis=1)

    X, Y, Paths = [], [], []
    for path, emotion in zip(df['path'], df['emotions']):
        features = get_features(path)
        for variant in features:
            X.append(variant)
            Y.append(emotion)
            Paths.append(path.split("/")[-1])
            
            
    features = pd.DataFrame(X)
    features['label'] = Y
    features['path'] = Paths
    features = features.fillna(0) # remove NaN
    features.to_csv('features.csv', index=False)

    X = features.drop(columns=['label', 'path'])
    Y = features['label']

    # We need to autoencode the labels for our classifier
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(Y.to_numpy().reshape(-1,1)).toarray()

    # Save encoder to use it later for predictions
    with open("encoder", "wb") as f:
        pickle.dump(encoder, f)
        f.close()

    # splitting data in train, validation and test
    x_train, x_test, y_train, y_test = train_test_split(X, Y,test_size=0.1, random_state=4, shuffle=True)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=4, test_size=0.2, shuffle=True)

    # scaling data and making it compatible with the model
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_val=scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    x_val = np.expand_dims(x_val, axis=2)

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_features_from_single_file(file_path):
    # Load the features from the saved CSV
    try:
        df = pd.read_csv('features.csv')
    except FileNotFoundError:
        print(f"Could not find extracted features. Did you train the model?")
        return None, None

    # Find the row for the given file path
    row = df.loc[df['path'] == file_path]

    # If row is empty, file path was not found in the CSV
    if row.empty:
        print(f"Could not find features for file {file_path} in features.csv, did you train the model?")
        return None, None

    # Drop the 'label' and 'path' columns
    X = row.drop(columns=['label', 'path'])
    Y = row['label']

    # Return the remaining feature columns as a numpy array
    return X.to_numpy(), Y.to_numpy()
