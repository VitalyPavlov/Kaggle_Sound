import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import label_ranking_average_precision_score
from keras.models import Model
from keras.callbacks import *
from keras.layers import *
from keras import optimizers
from bin.train_data_generator import DataGenerator
from bin.test_data_generator import TestDataGenerator
from bin.attention_layer import Attention


SEED = 42
N_FOLD = 5
DATASET = 'melspectrogram'  # mcc
NUM_CLASSES = 80
PATH = "/Users/vivpavlov/Documents/PythonScripts/Kaggle/Sound"


def label_encoding(y, label_columns):
    y_encoded = np.zeros((len(y), NUM_CLASSES)).astype(int)
    for i, row in enumerate(y):
        for label in row.split(','):
            idx = label_columns.index(label)
            y_encoded[i, idx] = 1
    return y_encoded


def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0,
        scores[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap


def lstm_layers(inp_shape):
    inp = Input(shape=(inp_shape[0], inp_shape[1],))

    x = Bidirectional(LSTM(128, return_sequences=True), merge_mode='concat')(inp)
    x = Bidirectional(LSTM(128, return_sequences=True), merge_mode='concat')(x)
    x = Attention(inp_shape[0])(x)

    x = Dense(128, activation="relu")(x)
    x = Dense(80, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    return model


def lstm_model(train_index, y_encoded, noisy_index, y_encoded_noisy):
    x_train, x_val, y_train, y_val = train_test_split(train_index, y_encoded,
                                                      test_size=0.2, random_state=SEED)

    params = {
        'n_classes': 80,
        'normalized': True,
        'dim': (127, 128)
    }

    train_gen = DataGenerator(x_train, y_encoded, batch_size=128, shuffle=True,
                              base_dir='../data/train_data/X_train_audio_to_%s_' % DATASET, **params)
    valid_gen = DataGenerator(x_val, y_encoded, batch_size=1, shuffle=False,
                              base_dir='../data/train_data/X_train_audio_to_%s_' % DATASET, **params)
    noisy_gen = DataGenerator(noisy_index, y_encoded_noisy, batch_size=128, shuffle=True,
                              base_dir='../data/noisy_data/X_noisy_audio_to_%s_' % DATASET, **params)

    K.clear_session()
    model = lstm_layers(params['dim'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy'])
    ckpt = ModelCheckpoint('../data/weights_n.h5',
                           save_best_only=True,
                           save_weights_only=True,
                           verbose=1,
                           monitor='val_categorical_crossentropy',
                           mode='min')
    model.fit_generator(train_gen, epochs=30, verbose=1,
                        use_multiprocessing=False,
                        validation_data=valid_gen,
                        callbacks=[ckpt])

    K.clear_session()
    model = lstm_layers(params['dim'])
    optimizer = optimizers.Adam(0.0000002)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_crossentropy'])

    ckpt = ModelCheckpoint('../data/weights_n.h5',
                           save_best_only=True,
                           save_weights_only=True,
                           verbose=1,
                           monitor='val_categorical_crossentropy',
                           mode='min')

    model.load_weights('../data/weights_n.h5')

    model.fit_generator(noisy_gen, epochs=5, verbose=1,
                        use_multiprocessing=False,
                        validation_data=valid_gen,
                        callbacks=[ckpt])

    model.load_weights('../data/weights_n.h5')

    preds_val = model.predict_generator(valid_gen, use_multiprocessing=True, verbose=1)
    print(preds_val.shape, y_val.shape)
    loss = calculate_overall_lwlrap_sklearn(y_val, preds_val)
    print('Overall_lwlrap: {:.5f}'.format(loss))

    return preds_val, y_val, loss, model


def fit():
    test = pd.read_csv(PATH + '/sample_submission.csv')
    label_columns = list(test.columns[1:])

    list_dir = os.listdir('../data/train_data')
    list_dir.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    list_dir = [x for x in list_dir if x.split('_')[4] == DATASET]
    train_index = np.arange(len(list_dir))

    y = np.load('../data/y_train.npy')
    y_encoded = label_encoding(y, label_columns)

    list_dir = os.listdir('../data/noisy_data')
    list_dir.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    list_dir = [x for x in list_dir if x.split('_')[4] == DATASET]
    noisy_index = np.arange(len(list_dir))

    y_noisy = np.load('../data/y_noisy.npy')
    y_encoded_noisy = label_encoding(y_noisy, label_columns)

    preds_val, y_val, loss, model = lstm_model(train_index, y_encoded, noisy_index, y_encoded_noisy)

    return model


def predict(model):
    list_dir = os.listdir('../data/test_data')
    list_dir.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    list_dir = [x for x in list_dir if x.split('_')[4] == DATASET]
    test_index = np.arange(len(list_dir))

    params = {
        'base_dir': '../data/test_data/X_test_audio_to_%s_' % DATASET,
        'normalized': True,
        'dim': (127, 128)
    }

    length = []
    for ID in test_index:
        x = np.load('../data/test_data/X_test_audio_to_%s_' % DATASET + str(ID) + '.npy')
        length.append(x.shape[0] // params['dim'][0])

    count = []
    for ind, el in enumerate(length):
        count.extend([ind] * el)

    test_gen = TestDataGenerator(test_index, count, shuffle=False, **params)

    preds = model.predict_generator(test_gen, use_multiprocessing=True, verbose=1)
    preds = pd.DataFrame(preds)

    preds['count'] = count
    preds = preds.groupby(by=count).mean()

    del preds['count']

    test = pd.read_csv(PATH + '/sample_submission.csv')
    test.iloc[:, 1:] = preds.values
    test.to_csv('../data/submission.csv', index=False)


def main():
    model = fit()
    predict(model)


if __name__ == "__main__":
    main()
