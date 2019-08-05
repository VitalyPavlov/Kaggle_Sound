import numpy as np
import pandas as pd
import librosa
import gc
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import skew, kurtosis


SAMPLE_RATE = 44100
JOBS = 3
PATH = "/Users/vivpavlov/Documents/PythonScripts/Kaggle/Sound"


def part_time_features(data, n=2):
    X_ = []
    for i in range(0, len(data), len(data) // n):
        X_.append(np.mean(data[i:i + len(data) // n]))
        X_.append(np.std(data[i:i + len(data) // n]))
        X_.append(np.min(data[i:i + len(data) // n]))
        X_.append(np.max(data[i:i + len(data) // n]))
        X_.append(skew(data[i:i + len(data) // n]))
        X_.append(kurtosis(data[i:i + len(data) // n]))
    return X_


def spectral_features(data, n_mfcc=20):
    spectral_features = [
        librosa.feature.spectral_centroid,
        librosa.feature.spectral_bandwidth,
        librosa.feature.spectral_contrast,
        librosa.feature.spectral_rolloff,
        librosa.feature.spectral_flatness,
        librosa.feature.zero_crossing_rate]
    try:
        M = librosa.feature.mfcc(data, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
        data_row = np.hstack((np.mean(M, axis=1), np.std(M, axis=1), np.min(M, axis=1),
                              np.max(M, axis=1), skew(M, axis=1), kurtosis(M, axis=1)))

        for feat in spectral_features:
            S = feat(data)[0]
            data_row = np.hstack((data_row, np.mean(S), np.std(S), np.min(S),
                                  np.max(S), skew(S), kurtosis(S)))
    except ImportError:
        return np.array([0] * 216)
    return data_row


def create_features(path):
    data, _ = librosa.load(path, sr=SAMPLE_RATE)
    lenght = len(data)
    data, _ = librosa.effects.trim(data)
    lenght_trim = len(data)
    splits = librosa.effects.split(data, top_db=40)
    if len(splits) > 1:
        data_split = np.concatenate([data[x[0]:x[1]] for x in splits])
        length_split = len(data_split)
    else:
        length_split = lenght

    ratio_trim = lenght_trim / lenght
    ratio_split = length_split / lenght

    abs_data = np.abs(data)
    diff_data = np.diff(data)

    X = [ratio_trim, ratio_split]
    for n in range(1, 2):
        X.extend(part_time_features(data, n))
        X.extend(part_time_features(abs_data, n))
        X.extend(part_time_features(diff_data, n))

    X.extend(spectral_features(data, n_mfcc=30))
    return X


def main():
    gc.enable()
    train_curated = pd.read_csv(PATH + '/train_curated.csv')
    test = pd.read_csv(PATH + '/sample_submission.csv')

    label_columns = list(test.columns[1:])
    label_mapping = dict((label, index) for index, label in enumerate(label_columns))

    # Train
    train_curated['path'] = PATH + '/train_curated/' + train_curated['fname']
    train_curated['num_labels'] = train_curated['labels'].apply(lambda x: len(x.split(',')))

    mask = train_curated.num_labels == 2
    first_label = train_curated['labels'].apply(lambda x: label_mapping[x.split(',')[0]]).values
    second_label = train_curated[mask]['labels'].apply(lambda x: label_mapping[x.split(',')[1]]).values

    train_meta = train_curated.append(train_curated[mask], ignore_index=True)
    train_meta['target'] = np.concatenate((first_label, second_label), axis=0)

    train_data = Parallel(n_jobs=-JOBS)(delayed(create_features)(fn) for fn in tqdm(train_meta['path'].values))

    np.save('../data/train_data.npy', train_data)
    train_meta.to_csv('../data/train_meta.csv')

    del train_curated, train_data, train_meta
    gc.collect()

    # Test
    test_path = PATH + '/test/' + test['fname']
    x_test = Parallel(n_jobs=JOBS)(delayed(create_features)(fn) for fn in tqdm(test_path.values))
    np.save('../data/test_data.npy', x_test)

    del test
    gc.collect()


if __name__ == '__main__':
    main()
