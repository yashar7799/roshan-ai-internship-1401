from typing import Tuple
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPool2D, BatchNormalization, Activation
from tensorflow.keras.models import Model






class Model1():

    """
    The model architecture from ..... paper.
    """

    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (120, 120, 3),
                 num_classes: int = 34,
                 pre_trained: bool = False,
                 model_path: str = None,
                 dropout: float = 0.5):

        """
        :param input_shape: input shape for the model to be built with
        :param num_classes: number of classes in the classification problem
        """

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pre_trained = pre_trained
        self.model_path = model_path
        self.dropout = dropout

    def conv_block(self, input):
        
        conv = Conv2D(4, (5, 5), (1, 1), padding='same')(input)
        bn = BatchNormalization()(conv)
        af = Activation(tf.nn.crelu)(bn)

        return af

    def conv_maxpool_block(self, input):

        conv = Conv2D(8, (5, 5), (1, 1), padding='same')(input)
        bn = BatchNormalization()(conv)
        af = Activation(tf.nn.crelu)(bn)
        conv = Conv2D(16, (5, 5), (2, 2), padding='same')(af)
        bn = BatchNormalization()(conv)
        af = Activation(tf.nn.crelu)(bn)
        pool = MaxPool2D((2, 2), (1, 1), padding='same')(af)
        
        return pool


    def get_model(self):

        """ model loader """

        input = Input(self.input_shape)

        conv = self.conv_block(input)
        conv = self.conv_block(conv)
        conv = self.conv_block(conv)
        conv = self.conv_block(conv)

        conv_maxpool = self.conv_maxpool_block(conv)
        conv_maxpool = self.conv_maxpool_block(conv_maxpool)
        conv_maxpool = self.conv_maxpool_block(conv_maxpool)
        conv_maxpool = self.conv_maxpool_block(conv_maxpool)
        conv_maxpool = self.conv_maxpool_block(conv_maxpool)
        conv_maxpool = self.conv_maxpool_block(conv_maxpool)

        flatten = Flatten()(conv_maxpool)
        dropout = Dropout(self.dropout)(flatten)

        dense = Dense(40, tf.nn.crelu)(dropout)
        dropout = Dropout(self.dropout)(dense)

        dense = Dense(20, tf.nn.crelu)(dropout)
        dropout = Dropout(self.dropout)(dense)

        classifier = Dense(self.num_classes, 'softmax')(dropout)

        model = Model(inputs=input, outputs=classifier)

        if self.pre_trained:
            model.load_weights(self.model_path)
        
        return model