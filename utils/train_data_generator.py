import numpy as np
import keras
import random


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(127, 128), n_channels=1,
                 n_classes=80, shuffle=True, base_dir='../data/train_data/X_train_audio_to_mcc_',
                 normalized=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.base_dir = base_dir
        self.normalized = normalized
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim)) # self.n_channels
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x = np.load(self.base_dir + str(ID) + '.npy')
            n = x.shape[0]
            crop = random.randint(0, n - self.dim[0])
            X[i, ] = x[crop: crop + self.dim[0], :]

            if self.normalized:
                max_data = np.max(X[i, ])
                min_data = np.min(X[i, ])
                X[i, ] = (X[i, ] - min_data) / (max_data - min_data + 1e-6)

            # Store class
            y[i, ] = self.labels[ID]

        return X, y
