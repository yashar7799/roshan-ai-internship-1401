"""
This module load user selected model.

this modules contain following Functions:
    -   load_model
"""

from .mobilenet import MobileNetV2
from .efficientnet import EfficientNetB0
from .resnet import ResNet18, ResNet50, ResNet50V2
from .densenet import DenseNet121
from .inception import InceptionV3
from .nasnet import NASNetMobile, NASNetLarge
from .vgg import VGG16, VGG19
from .xception import Xception
from .from_scratch import Model1

MODELS = dict(
    densenet121=DenseNet121,
    efficientnet_b0=EfficientNetB0,
    inception_v3=InceptionV3,
    mobilenet_v2=MobileNetV2,
    nasnet_mobile=NASNetMobile,
    nasnet_large=NASNetLarge,
    resnet18=ResNet18,
    resnet50=ResNet50,
    resnet50v2=ResNet50V2,
    vgg16=VGG16,
    vgg19=VGG19,
    xception=Xception,
    model1 = Model1
)


def load_model(model_name, **kwargs):
    """ Get model
       Parameters
       ----------
       model_name  string, user selected model name

       Returns
       -------
       model     tensorflow model instance
       """
    return MODELS[model_name](**kwargs).get_model()
