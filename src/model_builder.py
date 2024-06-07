import io
import logging
import tensorflow as tf

from contextlib import redirect_stdout
from functools import partial
from omegaconf import DictConfig

from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Conv2D, \
    AveragePooling2D, MaxPool2D, Flatten, Input  # type: ignore

from .helpers import log_config_error

log = logging.getLogger(__name__)


def build_model(cfg: DictConfig, **kwargs) -> tf.keras.Model:
    """
    Builds a keras model based on the specified type.
    Args:
        cfg (DictConfig): The model configuration

    Returns:
        tf.keras.Model: The final model
    """

    architecture = cfg.architecture.lower()

    if architecture in ['cnn', 'conv']:
        model = build_cnn(cfg, **kwargs)

    log.info("----------------------Model Summary----------------------\n\n")

    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()

    model_summary = stream.getvalue()
    log.info(model_summary)

    return model


def build_cnn(cfg: DictConfig,
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

    try:
        num_conv_layers = int(cfg.num_conv_layers)
        num_dense_layers = int(cfg.num_dense_layers)

    except TypeError:
        log_config_error(
            'Number of Convolutional/Dense layers must be a number, got:\n'
            f'Type of N Conv Layers:\t{type(num_conv_layers)}\n'
            f'Type of N Dense Layers\t{type(num_dense_layers)}'
        )

    filters = cfg.filters
    dense_neurons = cfg.dense_neurons

    DefaultConv = _get_conv_partial(
        kernel_size=cfg.conv_kernel_size,
        padding=cfg.conv_padding,
        activation=cfg.conv_activation,
        kernel_initializer=cfg.initializer,
        strides=cfg.conv_strides
        )

    DefaultPool = _get_pool_partial(
        pool_size=cfg.pool_size,
        padding=cfg.pool_padding,
        strides=cfg.pool_strides
    )

    DefaultDense = _get_dense_partial(
        activation=cfg.dense_activation,
        kernel_initializer=cfg.initializer,
    )

    # Input layer
    inputs = Input(shape=input_shape, name="InputLayer")
    x = DefaultConv(
        filters=filters[0], kernel_size=7, name="InputConv")(inputs)
    x = DefaultPool(name="InputPool")(x)

    # Convolutional layers
    for i in range(num_conv_layers):
        x = DefaultConv(filters=filters[i+1], name=f'Conv{i+1}-1')(x)

        if cfg.pair_convs:
            x = DefaultConv(filters=filters[i+1], name=f'Conv{i+1}-2')(x)

        x = DefaultPool(name=f"Pool{i+1}")(x)

    x = Flatten(name="Flatten")(x)

    # Dense (fully connected) layers
    for i in range(num_dense_layers):
        x = DefaultDense(
                  units=dense_neurons[i],
                  name=f'Dense{i+1}')(x)

        if cfg.pair_dropout:
            x = Dropout(cfg.dropout,
                        name=f"Dropout{i+1}")(x)

    if not cfg.pair_dropout:
        x = Dropout(cfg.dropout, name="Dropout1")(x)

    class_out = Dense(
        num_classes, activation='softmax', name="OutputClasses"
        )(x)

    if cfg.add_regressor:
        regressor_out = Dense(1, activation='sigmoid', name='Regressor')(x)
        model = Model(inputs=inputs, outputs=[class_out, regressor_out])

    else:
        model = Model(inputs=inputs, outputs=class_out)

    return model


def _get_dense_partial(**kwargs):
    return partial(Dense, **kwargs)


def _get_conv_partial(**kwargs):
    return partial(Conv2D, **kwargs)


def _get_pool_partial(pool_type: str = "avg", **kwargs):
    pool_type = pool_type.lower()

    if pool_type in ['avg', 'average']:
        return partial(AveragePooling2D, **kwargs)

    if pool_type in ['max', 'maximum']:
        return partial(MaxPool2D, **kwargs)

    log_config_error(f'Invalid pool type: {pool_type}')
