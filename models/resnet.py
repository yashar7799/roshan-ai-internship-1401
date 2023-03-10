from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import resnet, resnet_v2
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
import math
from tensorflow import keras
from tensorflow.keras import layers



class ResNet18():

    """
    The ResNet18 model
    """

    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (256, 256, 1),
                 num_classes: int = 10,
                 pre_trained: bool = False,
                 model_path: str = None):
        """
        :param model_path: where the model is located
        :param input_shape: input shape for the model to be built with
        :param num_classes: number of classes in the classification problem
        """
        self.model_path = model_path
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pre_trained = pre_trained
        self.model = self.get_model()

    def get_model(self):
        """ model loader """
        res = _Resnet18(self.input_shape, self.num_classes)
        print("get resnet model...")
        model = res.get_model()

        if self.pre_trained:
            model.load_weights(self.model_path)

        return model



class ResNet50():

    """
    The ResNet50 model
    """

    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (256, 256, 1),
                 num_classes: int = 10,
                 pre_trained: bool = False,
                 model_path: str = None,
                 imagenet_weights:bool=False):

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

    def get_model(self):
        """ model loader """

        if self.imagenet_weights:
            weights='imagenet'
        else:
            weights=None

        res = resnet.ResNet50(input_shape= (self.input_shape[0], self.input_shape[1], self.input_shape[2]) ,  include_top=False, pooling='max', weights=weights)
        model = Sequential()
        model.add(Conv2D(3, 1, activation='relu', padding='same', input_shape=self.input_shape))
        model.add(res)
        # model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='sigmoid'))

        if self.pre_trained:
            model.load_weights(self.model_path)

        return model





class ResNet50V2():

    """
    The ResNet50V2 model
    """

    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (256, 256, 1),
                 num_classes: int = 10,
                 pre_trained: bool = False,
                 model_path: str = None,
                 imagenet_weights:bool=False):

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

    def get_model(self):
        """ model loader """

        if self.imagenet_weights:
            weights='imagenet'
        else:
            weights=None

        res = resnet_v2.ResNet50V2(input_shape= (self.input_shape[0], self.input_shape[1], self.input_shape[2]) ,  include_top=False, pooling='max', weights=weights)
        model = Sequential()
        model.add(Conv2D(3, 1, activation='relu', padding='same', input_shape=self.input_shape))
        model.add(res)
        # model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='sigmoid'))

        if self.pre_trained:
            model.load_weights(self.model_path)

        return model


















class _Resnet18:
    def __init__(self, image_size, n_classes=10):
        self.input_shape = (image_size[0], image_size[1], image_size[2])
        self.n_classes = n_classes

    def get_model(self) -> keras.Model:
        kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

        def conv3x3(x, out_planes, stride=1, name=None):
            x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
            return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False,
                                kernel_initializer=kaiming_normal, name=name)(x)


        def basic_block(x, planes, stride=1, downsample=None, name=None):
            identity = x

            out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
            out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
            out = layers.ReLU(name=f'{name}.relu1')(out)

            out = conv3x3(out, planes, name=f'{name}.conv2')
            out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

            if downsample is not None:
                for layer in downsample:
                    identity = layer(identity)

            out = layers.Add(name=f'{name}.add')([identity, out])
            out = layers.ReLU(name=f'{name}.relu2')(out)

            return out


        def make_layer(x, planes, blocks, stride=1, name=None):
            downsample = None
            inplanes = x.shape[3]
            if stride != 1 or inplanes != planes:
                downsample = [
                    layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False,
                                kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
                    layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
                ]

            x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
            for i in range(1, blocks):
                x = basic_block(x, planes, name=f'{name}.{i}')

            return x


        def resnet(x, blocks_per_layer, num_classes=self.n_classes):
            x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
            x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal,
                            name='conv1')(x)
            x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
            x = layers.ReLU(name='relu1')(x)
            x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
            x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

            x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
            x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
            x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
            x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

            x = layers.GlobalAveragePooling2D(name='avgpool')(x)
            initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
            x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)
            x = layers.Softmax()(x)
            return x


        inputs = keras.Input(shape=self.input_shape)
        outputs = resnet(inputs, [2, 2, 2, 2], self.n_classes)
        model = keras.Model(inputs, outputs)
        return model



