from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import nasnet
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten


class NASNetMobile():

    """
    The NASNetMobile model
    """

    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (120, 120, 3),
                 num_classes: int = 34,
                 pre_trained: bool = False,
                 model_path: str = None,
                 dropout: float = 0.4,
                 imagenet_weights: bool=False):

        """
        :param model_path: where the model is located
        :param input_shape: input shape for the model to be built with
        :param num_classes: number of classes in the classification problem
        """

        self.model_path = model_path
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pre_trained = pre_trained
        self.imagenet_weights = imagenet_weights
        self.dropout = dropout

    def get_model(self):
        """ model loader """

        if self.imagenet_weights:
            weights='imagenet'
        else:
            weights=None

        dense = nasnet.NASNetMobile(input_shape= (self.input_shape[0], self.input_shape[1], self.input_shape[2]) ,  include_top=False, pooling='max', weights=weights)
        model = Sequential()
        model.add(Conv2D(3, 1, activation='relu', padding='same', input_shape=self.input_shape))
        model.add(dense)
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.num_classes, activation='sigmoid'))

        if self.pre_trained:
            model.load_weights(self.model_path)

        return model






class NASNetLarge():

    """
    The NASNetLarge model
    """

    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (120, 120, 3),
                 num_classes: int = 34,
                 pre_trained: bool = False,
                 model_path: str = None,
                 dropout: float = 0.4,
                 imagenet_weights: bool=False):

        """
        :param model_path: where the model is located
        :param input_shape: input shape for the model to be built with
        :param num_classes: number of classes in the classification problem
        """

        self.model_path = model_path
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pre_trained = pre_trained
        self.imagenet_weights = imagenet_weights
        self.dropout = dropout

    def get_model(self):
        """ model loader """

        if self.imagenet_weights:
            weights='imagenet'
        else:
            weights=None

        dense = nasnet.NASNetLarge(input_shape= (self.input_shape[0], self.input_shape[1], self.input_shape[2]) ,  include_top=False, pooling='max', weights=weights)
        model = Sequential()
        model.add(Conv2D(3, 1, activation='relu', padding='same', input_shape=self.input_shape))
        model.add(dense)
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.num_classes, activation='sigmoid'))

        if self.pre_trained:
            model.load_weights(self.model_path)

        return model