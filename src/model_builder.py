import tensorflow as tf

from functools import partial
from omegaconf import DictConfig

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

    elif architecture == 'alexnet':
        model = build_alexnet(cfg, **kwargs)

    return model


def build_alexnet(cfg: DictConfig,
                  input_shape: tf.TensorShape = tf.TensorShape([100, 100, 55]),
                  num_classes: int = 3) -> tf.keras.Model:
    """
    Builds AlexNet-like model for class prediction
    Args:
        cfg (DictConfig): The model config
        input_shape (tf.TensorShape, optional): The shape of the image inputs
        to the CNN. Defaults to tf.TensorShape([100, 100, 55]).
        num_classes (int, optional): The number of classes to output.
        Defaults to 3.

    Returns:
        tf.keras.Model: The AlexNet model.
    """

    if type(input_shape) is list:
        input_shape = tf.TensorShape(input_shape)

    layers = [layer.lower() for layer in cfg.layers]

    for layer in layers:
        assert layer in ['conv2d', 'maxpooling2d', 'averagepooling2d',
                         'dense', 'flatten']

    num_conv_layers = len([layer for layer in layers if 'conv' in layer])
    num_pool_layers = len([layer for layer in layers if 'pool' in layer])
    num_dense_layers = len([layer for layer in layers if 'dense' in layer])

    assert len(cfg.filters) == len(cfg.conv_kernels) == len(cfg.conv_pads) == \
        len(cfg.conv_strides) == num_conv_layers

    assert len(cfg.pool_sizes) == len(cfg.pool_pads) == \
        len(cfg.pool_strides) == num_pool_layers

    assert len(cfg.dense_neurons) == len(cfg.dropouts) == num_dense_layers

    inputs = Input(shape=input_shape, name="InputLayer")
    x = inputs

    conv_layer = 0
    pool_layer = 0
    flatten_layer = 0
    dense_layer = 0

    print(layers)

    for layer in layers:
        if layer == 'conv2d':
            x = Conv2D(filters=cfg.filters[conv_layer],
                       kernel_size=cfg.conv_kernels[conv_layer],
                       activation=cfg.activation,
                       strides=cfg.conv_strides[conv_layer],
                       padding=cfg.conv_pads[conv_layer],
                       kernel_initializer=cfg.initializer,
                       name=f'Conv{conv_layer}')(x)
            conv_layer += 1

        elif layer == 'maxpooling2d':
            x = MaxPooling2D(pool_size=cfg.pool_sizes[pool_layer],
                             strides=cfg.pool_strides[pool_layer],
                             padding=cfg.pool_pads[pool_layer],
                             name=f'MaxPool{pool_layer}')(x)

            pool_layer += 1

        elif layer == 'averagepooling2d':
            x = AveragePooling2D(pool_size=cfg.pool_sizes[pool_layer],
                                 strides=cfg.pool_strides[pool_layer],
                                 padding=cfg.pool_pads[pool_layer],
                                 name=f'AveragePool{pool_layer}')(x)

            pool_layer += 1

        elif layer == 'flatten':
            x = Flatten(name=f'Flatten{flatten_layer}')(x)

            flatten_layer += 1

        elif layer == 'dense':
            x = Dense(cfg.dense_neurons[dense_layer],
                      activation=cfg.activation,
                      name=f'Dense{dense_layer}')(x)

            x = Dropout(cfg.dropouts[dense_layer],
                        name=f'Dropout{dense_layer}')(x)

            dense_layer += 1

        else:
            raise RuntimeError('Weird Error')

    class_out = Dense(num_classes, activation='softmax',
                      name='OutputLayer')(x)

    model = Model(inputs=inputs, outputs=class_out)
    print(model.summary())

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
        raise TypeError(
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
        return partial(MaxPooling2D, **kwargs)

    raise ValueError(f'Invalid pool type: {pool_type}')


def _get_optimizer(optimizer_name, **kwargs):
    return OPTIMIZERS.get(optimizer_name)(**kwargs)


def _get_loss(loss_name, **kwargs):
    return LOSSES.get(loss_name)(**kwargs)


def _get_metric(metric_name, **kwargs):
    return METRICS.get(metric_name)(**kwargs)
