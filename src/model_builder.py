import tensorflow as tf

from functools import partial
from omegaconf import DictConfig, OmegaConf

from tensorflow import keras
from keras.models import Model  # type: ignore
from keras.layers import Dense, Dropout, Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Input  # type: ignore

from icecream import ic

OPTIMIZERS = {
    'adam': tf.keras.optimizers.Adam,
    'sgd': tf.keras.optimizers.SGD,
}

LOSSES = {
    'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy
}

METRICS = {
    'categorical_accuracy': tf.keras.metrics.CategoricalAccuracy
}

LAYERS = {
    'conv2d': Conv2D,
    'maxpooling2d': MaxPooling2D,
    'averagepooling2d': AveragePooling2D,
    'dense': Dense,
    'flatten': Flatten,
    'dropout': Dropout,
}


def build_model(cfg: DictConfig,
                input_shape: tf.TensorShape = tf.TensorShape([100, 100, 55]),
                num_classes: int = 3) -> tf.keras.Model:
    """
    Builds a CNN model based on the model config
    Args:
        cfg (DictConfig): The model config
        input_shape (tf.TensorShape, optional): The shape of the image inputs
        to the CNN. Defaults to tf.TensorShape([100, 100, 55]).
        num_classes (int, optional): The number of classes to output.
        Defaults to 3.

    Returns:
        tf.keras.Model: The CNN model.
    """

    if type(input_shape) is list:
        input_shape = tf.TensorShape(input_shape)

    default_layers = OmegaConf.to_container(cfg.architecture.defaults)

    for layer in default_layers.keys():
        assert layer.lower() in LAYERS, f'Invalid layer type provided: {layer}'

        DefaultLayer = _get_partial(layer, **default_layers[layer])
        default_layers[layer] = DefaultLayer

    default_layers = {k.lower(): v for k, v in default_layers.items()}

    layers = OmegaConf.to_container(cfg.architecture.layers)

    inputs = Input(shape=input_shape, name="InputLayer")
    x = inputs

    for layer in layers.keys():
        layer_type = layers[layer]['type'].lower()
        del layers[layer]['type']

        assert layer_type in LAYERS, f'Invalid layer type provided: {layer}'

        try:
            if layer_type in default_layers.keys():
                x = default_layers.get(layer_type)(name=layer,
                                                   **layers[layer])(x)

            else:
                x = LAYERS.get(layer_type)(name=layer,
                                           **layers[layer])(x)

        except Exception as e:
            raise Exception(f'Error creating layer: {layer}, {e}')

    class_out = Dense(num_classes,
                      activation='softmax',
                      name='OutputLayer')(x)

    model = Model(inputs=inputs, outputs=class_out)

    return model


def _get_partial(layer_type, **kwargs):
    layer_type = layer_type.lower()
    return partial(LAYERS.get(layer_type), **kwargs)


def _get_optimizer(optimizer_name, **kwargs):
    return OPTIMIZERS.get(optimizer_name)(**kwargs)


def _get_loss(loss_name, **kwargs):
    return LOSSES.get(loss_name)(**kwargs)


def _get_metric(metric_name, **kwargs):
    return METRICS.get(metric_name)(**kwargs)
