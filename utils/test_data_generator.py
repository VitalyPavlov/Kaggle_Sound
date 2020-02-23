import numpy as np
import keras


class TestDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, count, dim=(127, 128),
                 shuffle=True, base_dir='../data/test_data/X_test_audio_to_mcc_',
                 normalized=False):
        'Initialization'
        self.dim = dim
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.base_dir = base_dir
        self.normalized = normalized
        self.count = count
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X = self.__data_generation(self.list_IDs)
        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((len(self.count), *self.dim))  # self.n_channels

        # Generate data
        i = 0
        for ID in list_IDs_temp:
            # Store sample
            x = np.load(self.base_dir + str(ID) + '.npy')
            n = x.shape[0]
            for crop in range(0, n // self.dim[0]):
                X[i,] = x[crop * self.dim[0]: (crop + 1) * self.dim[0], :]

                if self.normalized:
                    max_data = np.max(X[i,])
                    min_data = np.min(X[i,])
                    X[i,] = (X[i,] - min_data) / (max_data - min_data + 1e-6)
                i += 1

        return X
