import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, Activation, Dense, concatenate, add, Layer, Lambda
from tensorflow.keras.layers import Conv3DTranspose, LeakyReLU, MaxPooling3D
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D, Reshape, Dense, multiply, Permute, UpSampling3D

def squeeze_excite_block(input: tf.Tensor, ratio: int = 8) -> tf.Tensor:
    ''' 
    Create a channel-wise squeeze-excite block

    Parameters:
    `input`: input tensor
    `ratio`: ratio filters compared to the input filters

    Returns: a keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, 1, filters)

    se = GlobalAveragePooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((4, 1, 2, 3))(se)

    x = multiply([init, se])
    return x
    
def build_survival_model3(
    input_shape,
    detection_model=None,
    num_classes=1,
    final_activation="sigmoid",
    l2_regularization = 0.0001,
    instance_norm = False,
    num_clinical_parameters = 0
    ):
    """
    Build a dual attention U-Net.

    input_shape: tuple of dimensions (including channel axis)
    final_activation: activation function of the final layer
    l2_regularization: kernel regularization
    instance_norm: apply instance normalization
    """

    def conv_layer(x, kernel_size, out_filters, strides=(1,1,1)):
        x = Conv3D(out_filters, kernel_size, 
                strides             = strides,
                padding             = 'same',
                kernel_regularizer  = l2(l2_regularization), 
                kernel_initializer  = 'he_normal',
                use_bias            = False
                )(x)
        return x
    
    in_defaults = {
        "axis": -1,
        "center": True, 
        "scale": True,
        "beta_initializer": "random_uniform",
        "gamma_initializer": "random_uniform"
    }

    def conv_block(input, 
        out_filters, strides=(1,1,1), with_residual=False, 
        with_se=False, activation='relu'):
        # Strided convolution to convsample
        x = conv_layer(input, (3,3,3), out_filters, strides)
        x = Activation('relu')(x)

        # Unstrided convolution
        x = conv_layer(x, (3,3,3), out_filters)

        # Add a squeeze-excite block
        if with_se:
            se = squeeze_excite_block(x)
            x = add([x, se])
            
        # Add a residual connection using a 1x1x1 convolution with strides
        if with_residual:
            residual = conv_layer(input, (1,1,1), out_filters, strides)
            x = add([x, residual])
            
        if instance_norm:
            x = InstanceNormalization(**in_defaults)(x)
            
        if activation == 'leaky':
            x = LeakyReLU(alpha=.1)(x)
        else:
            x = Activation('relu')(x)
        
        # Activate whatever comes out of this
        return x

    # If we already have only one input, no need to combine anything
    image_inputs = Input(input_shape)
    if detection_model is not None:
        detection_output = detection_model(image_inputs)
        images = detection_output
    else:
        images = image_inputs

    time_input = Input(1)
    if num_clinical_parameters > 0:
        clinical_inputs = Input(num_clinical_parameters)
        inputs = [image_inputs, clinical_inputs, time_input]
    else:
        inputs = [image_inputs, time_input]

    # Downsampling
    conv1 = conv_block(detection_output, 16, strides=(2,2,1), with_residual=True, with_se=True)
    conv2 = conv_block(conv1, 32, strides=(2,2,1), with_residual=True, with_se=True) #72x72x18
    conv3 = conv_block(conv2, 64, strides=(2,2,2), with_residual=True, with_se=True) #36x36x18
    conv4 = conv_block(conv3, 128, strides=(2,2,2), with_residual=True, with_se=True) #18x18x9
    image_features = GlobalAveragePooling3D()(conv4)
    image_features = Dense(16, activation="relu")(image_features)
    
    # Combine image features with clinical features
    if num_clinical_parameters > 0:
        all_features = concatenate([image_features, clinical_inputs, time_input])
    else:
        all_features = concatenate([image_features, time_input])
    
    all_features = Dense(16, activation="relu")(all_features)
    
    # Reduce to single value
    upgrade_likelihood = Dense(1, activation="sigmoid")(all_features)
    
    # Model definition
    model = Model(inputs=inputs, outputs=upgrade_likelihood)
    return model
