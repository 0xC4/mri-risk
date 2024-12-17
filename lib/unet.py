import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, Activation, Dense, concatenate, add, Layer, Lambda
from tensorflow.keras.layers import Conv3DTranspose, LeakyReLU, MaxPooling3D
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import GlobalAveragePooling3D, Reshape, Dense, multiply, Permute, UpSampling3D

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
    

def build_dual_attention_unet(
    input_shape,
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
    inputs = Input(input_shape)

    # Downsampling
    conv1 = conv_block(inputs, 16)
    conv2 = conv_block(conv1, 32, strides=(2,2,1), with_residual=True, with_se=True) #72x72x18
    conv3 = conv_block(conv2, 64, strides=(2,2,1), with_residual=True, with_se=True) #36x36x18
    conv4 = conv_block(conv3, 128, strides=(2,2,2), with_residual=True, with_se=True) #18x18x9
    conv5 = conv_block(conv4, 256, strides=(2,2,2), with_residual=True, with_se=True) #9x9x9

    if num_clinical_parameters > 0.5:
        clinical_inputs = Input(num_clinical_parameters)
        target_shape = (12, 12, 6, num_clinical_parameters)
        clinical_reshaped = Reshape((1,1,1,num_clinical_parameters))(clinical_inputs)
        print(clinical_reshaped)
        clinical_reshaped = tf.keras.layers.concatenate([clinical_reshaped]*target_shape[0], axis=1)
        print(clinical_reshaped)
        clinical_reshaped = tf.keras.layers.concatenate([clinical_reshaped]*target_shape[1], axis=2)
        print(clinical_reshaped)
        clinical_reshaped = tf.keras.layers.concatenate([clinical_reshaped]*target_shape[2], axis=3)
        print(clinical_reshaped)
        #clin_reshaped = tf.reshape(clinical_inputs, (tf.shape(inputs)[0], 1,1,1,num_clinical_parameters))
        
        #print("CONV5shape:", conv5.shape)
        #clinical_params = tf.tile(clin_reshaped, (1, *conv5.shape[1:4], 1))
        conv5 = concatenate([conv5, clinical_reshaped], axis=-1)

        inputs = [inputs, clinical_inputs]
    
    # First upsampling sequence
    up1_1 = Conv3DTranspose(128, (3,3,3), strides=(2,2,2), padding='same')(conv5) #18x18x9
    up1_2 = Conv3DTranspose(128, (3,3,3), strides=(2,2,2), padding='same')(up1_1) #36x36x18
    up1_3 = Conv3DTranspose(128, (3,3,3), strides=(2,2,1), padding='same')(up1_2) #72x72x18
    bridge1 = concatenate([conv4, up1_1]) #18x18x9 (128+128=256)
    dec_conv_1 = conv_block(bridge1, 128, with_residual=True, with_se=True, activation='leaky') #18x18x9

    # Second upsampling sequence
    up2_1 = Conv3DTranspose(64, (3,3,3), strides=(2,2,2), padding='same')(dec_conv_1) # 36x36x18
    up2_2 = Conv3DTranspose(64, (3,3,3), strides=(2,2,1), padding='same')(up2_1) # 72x72x18
    bridge2 = concatenate([conv3, up1_2, up2_1]) # 36x36x18 (64+128+64=256)
    dec_conv_2 = conv_block(bridge2, 64, with_residual=True, with_se=True, activation='leaky')
    
    # Final upsampling sequence
    up3_1 = Conv3DTranspose(32, (3,3,3), strides=(2,2,1), padding='same')(dec_conv_2) # 72x72x18
    bridge3 = concatenate([conv2, up1_3, up2_2, up3_1]) # 72x72x18 (32+128+64+32=256)
    dec_conv_3 = conv_block(bridge3, 32, with_residual=True, with_se=True, activation='leaky')
    
    # Last upsampling to make heatmap
    up4_1 = Conv3DTranspose(16, (3,3,3), strides=(2,2,1), padding='same')(dec_conv_3) # 72x72x18
    dec_conv_4 = conv_block(up4_1, 16, with_residual=False, with_se=True, activation='leaky') #144x144x18 (16)

    # Reduce to a single output channel with a 1x1x1 convolution
    single_channel = Conv3D(num_classes, (1, 1, 1))(dec_conv_4)  

    # Apply sigmoid activation to get binary prediction per voxel
    act  = Activation(final_activation)(single_channel)
    
    # Model definition
    model = Model(inputs=inputs, outputs=act)
    return model

if __name__ == "__main__":
    test_model = build_dual_attention_unet((192, 192, 24, 3), 1, num_clinical_parameters=5)
    test_model.summary(line_length=120)