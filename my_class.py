from tensorflow.keras.utils import Sequence
import numpy as np
import os

class DataGenerator(Sequence):

    def __init__(self, file_list):
        """Constructor can be expanded,
           with batch size, dimentation etc.
        """
        self.file_list = file_list
        self.on_epoch_end()

    def __len__(self):
        'Take all batches in each iteration'
        return int(len(self.file_list))

    def __getitem__(self, index):
        'Get next batch'
        # Generate indexes of the batch
        indexes = self.indexes[index:(index+1)]
        print("getitem is called, index:",index)
        # single file
        file_list_temp = [self.file_list[k] for k in indexes]

        # Set of X_train and y_train
        X, y = self.__data_generation(file_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        print('on_epoch_end is called')
        self.indexes = np.arange(len(self.file_list))

    def __data_generation(self, file_list_temp):
        'Generates data containing batch_size samples'
        loc = os.path.abspath('.')
        # Generate data
        for ID in file_list_temp:
            x_file_path = os.path.join(loc,"workflow", "focustarget",str(ID))
            y_file_path = os.path.join(loc,"workflow",  "focuslabel",str(ID))

            # Store sample
            X = np.load(x_file_path)

            # Store class
            y = np.load(y_file_path)

        return X,y 




