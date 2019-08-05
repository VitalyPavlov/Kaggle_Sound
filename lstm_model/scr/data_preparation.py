import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import gc


SAMPLE_RATE = 44100
L = 2 * SAMPLE_RATE
HOP_LEGTH = 700
N_MELS = 128
N_FFT = 1024
JOBS = 3
PATH = "/Users/vivpavlov/Documents/PythonScripts/Kaggle/Sound"


def audio_to_melspectrogram(path, num_labels=1, place=None):
    samples, _ = librosa.load(path, sr=SAMPLE_RATE)
    samples, _ = librosa.effects.trim(samples)

    if place == 'first':
        samples = samples[: len(samples)//num_labels]
    elif place == 'last':
        samples = samples[(num_labels - 1) * len(samples)//num_labels:]

    splits = librosa.effects.split(samples, top_db=40)
    samples = np.concatenate([samples[x[0]:x[1]] for x in splits])

    if len(samples) < L:
        pad = (L - len(samples))//2
        samples = np.pad(samples, pad_width=(pad, L - len(samples) - pad), mode='constant', constant_values=(0, 0))

    spectrogram = librosa.feature.melspectrogram(samples,
                                                 sr=SAMPLE_RATE,
                                                 n_mels=N_MELS,
                                                 hop_length=HOP_LEGTH,
                                                 n_fft=N_FFT)

    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return np.transpose(spectrogram)


def audio_to_mcc(path, num_labels=1, place=None):
    samples, _ = librosa.load(path, sr=SAMPLE_RATE)
    samples, _ = librosa.effects.trim(samples)

    if place == 'first':
        samples = samples[: len(samples)//num_labels]
    elif place == 'last':
        samples = samples[(num_labels - 1) * len(samples)//num_labels:]

    splits = librosa.effects.split(samples, top_db=40)
    samples = np.concatenate([samples[x[0]:x[1]] for x in splits])

    if len(samples) < L:
        pad = (L - len(samples)) // 2
        samples = np.pad(samples, pad_width=(pad, L - len(samples) - pad), mode='constant', constant_values=(0, 0))

    spectrogram = librosa.feature.mfcc(samples, sr=SAMPLE_RATE, n_mfcc=30)
    return np.transpose(spectrogram)


def audio_to_stft(path, num_labels=1, place=None):
    samples, _ = librosa.load(path, sr=SAMPLE_RATE)
    samples, _ = librosa.effects.trim(samples)

    if place == 'first':
        samples = samples[: len(samples)//num_labels]
    elif place == 'last':
        samples = samples[(num_labels - 1) * len(samples)//num_labels:]

    splits = librosa.effects.split(samples, top_db=40)
    samples = np.concatenate([samples[x[0]:x[1]] for x in splits])

    if len(samples) < L:
        pad = (L - len(samples)) // 2
        samples = np.pad(samples, pad_width=(pad, L - len(samples) - pad), mode='constant', constant_values=(0, 0))

    window = np.hanning(N_FFT)
    samples = librosa.spectrum.stft(samples, n_fft=N_FFT, hop_length=HOP_LEGTH, window=window)
    samples = 2 * np.abs(samples) / np.sum(window)

    spectrogram = librosa.amplitude_to_db(samples, ref=np.max)
    return np.transpose(spectrogram)


def save_npy(df, function, name='train', label=0, place=None):
    for ind in tqdm(range(df.shape[0])):
        path = df.loc[ind, 'path']
        num_labels = 1
        X = function(path=path, num_labels=num_labels, place=place)
        np.save('../data/{0}_data/X_{0}_{1}_{2}.npy'.format(name, function.__name__, label), X)
        label += 1
        del X
        gc.collect()
    return label


def main():
    gc.enable()
    train_curated = pd.read_csv(PATH + '/train_curated.csv')
    train_noisy = pd.read_csv(PATH + '/train_noisy.csv')
    test = pd.read_csv(PATH + '/sample_submission.csv')

    functions = [audio_to_melspectrogram]  # audio_to_mcc, audio_to_stft

    # Train
    train_curated['path'] = PATH + '/train_curated/' + train_curated['fname']

    for function in functions:
        label = 0
        _ = save_npy(train_curated, function, name='train', label=label, place='first')

    y_train = train_curated['labels'].values
    np.save('../data/y_train.npy', y_train)

    del train_curated, y_train
    gc.collect()

    # Noisy
    train_noisy['path'] = PATH + '/train_noisy/' + train_noisy['fname']
    train_noisy['num_labels'] = train_noisy['labels'].apply(lambda x: len(x.split(',')))

    train_noisy = train_noisy[train_noisy.num_labels == 1].reset_index(drop=True)
    first_label = train_noisy['labels'].apply(lambda x: x.split(',')[0]).values

    tmp = pd.DataFrame(columns=train_noisy.columns)
    for el in set(first_label):
        tmp = tmp.append(train_noisy[first_label == el].iloc[:50])

    train_noisy = tmp.reset_index(drop=True)

    for function in functions:
        label = 0
        _ = save_npy(train_noisy, function, name='noisy', label=label, place='first')

    y_noisy = np.array(first_label)
    np.save('../data/y_noisy.npy', y_noisy)

    del train_noisy, y_noisy
    gc.collect()

    # Test
    test['path'] = PATH + '/test/' + test['fname']
    test['num_labels'] = 1
    for function in functions:
        label = 0
        _ = save_npy(test, function, name='test', label=label, place=None)

    del test
    gc.collect()


if __name__ == '__main__':
    main()
