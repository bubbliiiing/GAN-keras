import keras
from keras.models import *
from keras.layers import *
from keras import layers
import keras.backend as K
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

IMAGE_ORDERING = 'channels_last'
def one_side_pad( x ):
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    if IMAGE_ORDERING == 'channels_first':
        x = Lambda(lambda x : x[: , : , :-1 , :-1 ] )(x)
    elif IMAGE_ORDERING == 'channels_last':
        x = Lambda(lambda x : x[: , :-1 , :-1 , :  ] )(x)
    return x

def identity_block(input_tensor, kernel_size, filter_num, block):


    conv_name_base = 'res' + block + '_branch'
    in_name_base = 'in' + block + '_branch'
    # 1x1压缩
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(input_tensor)
    x = Conv2D(filter_num, (3, 3) , data_format=IMAGE_ORDERING , name=conv_name_base + '2a')(x)
    x = InstanceNormalization(axis=3,name=in_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(filter_num , (3, 3), data_format=IMAGE_ORDERING , name=conv_name_base + '2c')(x)
    x = InstanceNormalization(axis=3,name=in_name_base + '2c')(x)
    # 残差网络
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def get_resnet(input_height, input_width, channel):
    img_input = Input(shape=(input_height,input_width , 3 ))

    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING, strides=2)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), data_format=IMAGE_ORDERING, strides=2)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    for i in range(6):
        x = identity_block(x, 3, 256, block=str(i))

    x = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(x)
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    x = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(x)
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)    

    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(channel, (7, 7), data_format=IMAGE_ORDERING)(x)
    x = Activation('tanh')(x)  
    model = Model(img_input,x)
    return model
