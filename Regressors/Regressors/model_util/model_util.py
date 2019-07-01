
from keras import layers, regularizers


# Set Rule for random weight init. Default: glorot_uniform
weight_init = "glorot_uniform"


def encoding_layer(
                input_data,
                nfilter, kernal,
                stride=1, ndim=1,
                regularize= 0,
                activation=layers.advanced_activations.PReLU,
                act_kwargs={}
                ):

    x = layers.Conv2D(
        nfilter, (kernal, ndim),
        padding='same', strides=(stride, 1),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(regularize))(input_data)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = activation(**act_kwargs)(x)

    return x


def decoding_layer(
                input_data,
                nfilter, kernal,
                stride=1, ndim=1,
                drop=0,
                regularize=0,
                activation=layers.advanced_activations.PReLU,
                act_kwargs={}
                ):

    if drop:  # that droput does not appeare in the summary when value is 0
        input_data = layers.Dropout(drop)(input_data)

    x = layers.Conv2DTranspose(
        nfilter, (kernal, ndim),
        padding='same', strides=(stride, 1),
        kernel_initializer=weight_init,
        kernel_regularizer = regularizers.l2(regularize))(input_data)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = activation(**act_kwargs)(x)

    return x
