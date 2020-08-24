from keras.models import Sequential, Model
from keras import layers, regularizers


def build_image_ae(code_dim=128, regularization=0.0, dropout_rate=0.0, use_batchnorm=False, architecture='small_conv'):
        model = Sequential()
        if architecture=='small_conv':
            model.add(layers.Conv2D(kernel_size=(5,5), strides=(2,2), filters=128, activation='relu',
                                 input_shape=(32,32,1)))
            model.add(layers.Conv2D(kernel_size=(5,5), strides=(2,2), filters=256, activation='relu'))
            model.add(layers.Conv2D(kernel_size=(5,5), strides=(2,2), filters=512, activation='relu'))
            model.add(layers.Reshape((-1,)))
            model.add(layers.Dense(1024, activation='relu'))
            model.add(layers.Dense(1024, activation='relu'))
            model.add(layers.Dense(code_dim, activation='relu', name='im_code'))
        elif architecture=='dense_small_old':
            # This is for a 32x32 grayscale image
            # NEED TO REMOVE SOME OF THE HARDCODING HERE
            model.add(layers.Reshape((32*32,), input_shape=(32,32,1)))
            model.add(layers.Dense(128, activation='relu',  kernel_regularizer=regularizers.l2(regularization)))
            if use_batchnorm: model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=dropout_rate))
            model.add(layers.Dense(code_dim, activation='linear',  kernel_regularizer=regularizers.l2(regularization), name='im_code'))
            if use_batchnorm: model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=dropout_rate))
            model.add(layers.Dense(128, activation='relu',  kernel_regularizer=regularizers.l2(regularization)))
            if use_batchnorm: model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=dropout_rate))
            model.add(layers.Dense(1024, activation='sigmoid',  kernel_regularizer=regularizers.l2(regularization)))
            model.add(layers.Reshape((32, 32, 1)))
            model.add(layers.Dense(code_dim, activation='relu', name='im_code'))
        elif architecture=='large_rgb_conv':
            # This is for a 127x127 RGB image
            model.add(layers.Conv2D(input_shape=(127,127,3), kernel_size=(7,7), filters=96))
            model.add(layers.LeakyReLU(alpha=0.01))
            model.add(layers.MaxPool2D())
            
            model.add(layers.Conv2D(kernel_size=(3,3), filters=128))
            model.add(layers.LeakyReLU(alpha=0.01))
            model.add(layers.MaxPool2D())
            
            model.add(layers.Conv2D(kernel_size=(3,3), filters=256))
            model.add(layers.LeakyReLU(alpha=0.01))
            model.add(layers.MaxPool2D())
            
            model.add(layers.Conv2D(kernel_size=(3,3), filters=256))
            model.add(layers.LeakyReLU(alpha=0.01))
            model.add(layers.MaxPool2D())
            
            model.add(layers.Conv2D(kernel_size=(3,3), filters=256))
            model.add(layers.LeakyReLU(alpha=0.01))
            model.add(layers.MaxPool2D())
            
            #model.add(layers.Conv2D(kernel_size=(3,3), filters=256))
            #model.add(layers.LeakyReLU(alpha=0.01))
            #model.add(layers.MaxPool2D())
            
            model.add(layers.Reshape((-1,)))
            model.add(layers.Dense(code_dim, name='im_code', activation='relu'))
            print(model.summary())
            """
            Below was my impementation, swiching to R2N2
            base_filter_count = 16
            kernel_size = (3,3)
            stride_size = (2,2) 
            model.add(layers.Conv2D(input_shape=(128,128,3), filters=base_filter_count, kernel_size=(7,7),
                                    strides=stride_size))  # Maybe up strides?
            model.add(layers.LeakyReLU())
            model.add(layers.Conv2D(filters=2*base_filter_count, kernel_size=kernel_size, strides=stride_size))
            model.add(layers.LeakyReLU())
            model.add(layers.Conv2D(filters=4*base_filter_count, kernel_size=kernel_size, strides=stride_size))
            model.add(layers.LeakyReLU())
            model.add(layers.Conv2D(filters=8*base_filter_count, kernel_size=kernel_size, strides=stride_size))
            model.add(layers.LeakyReLU())
            model.add(layers.Conv2D(filters=16*base_filter_count, kernel_size=kernel_size, strides=stride_size))
            model.add(layers.LeakyReLU())
            model.add(layers.Reshape((-1,)))
            model.add(layers.Dense(code_dim, name='im_code', activation='relu'))
            print(model.summary()
            """
        return model


class ImageAutoencoder():
    
    def __init__(self,code_dim=128, image_dim=32, kernel_size=(5,5), filter_base_count=32,
                  strides=(2,2)):
        super(ImageAutoencoder, self).__init__()
        self.code_dim = code_dim
        self.image_dim = image_dim
        self.kernel_size = kernel_size
        self.filter_base_count = filter_base_count
        self.strides = strides
        self._build_self()
 
    def _build_self(self):
        self.model = Sequential()
        # NEED TO REMOVE SOME OF THE HARDCODING HERE
        self.model.add(layers.Reshape((32*32,), input_shape=(32,32,1)))
        # self.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dense(128, activation='relu'))
        # self.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dense(1024, activation='sigmoid'))
        self.model.add(layers.Reshape((32, 32, 1)))
        print(self.model.summary())
        return 
    
    def save_model(self, savepath):
        self.model.save(savepath)

        """
        self.add(layers.Conv2D(input_shape=(32,32,1), filters=self.filter_base_count,
                                   kernel_size=self.kernel_size,
                                   strides=self.strides,
                                   padding='same'))
        self.add(layers.Conv2D(2*self.filter_base_count,
                                   self.kernel_size,
                                   strides=self.strides,
                                   padding='same'))
        self.add(layers.Conv2D(4*self.filter_base_count,
                                   self.kernel_size,
                                   strides=self.strides,
                                   padding='same'))
        self.add(layers.Reshape((-1, 128*4*4)))
        self.add(layers.Dense(self.code_dim, activation='relu'))

        self.add(layers.Dense(128*4*4, activation='relu'))
        self.add(layers.Reshape((4, 4, 128)))
        self.add(layers.Conv2DTranspose(4*self.filter_base_count,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same'))
        self.add(layers.Conv2DTranspose(2*self.filter_base_count,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same'))
        self.add(layers.Conv2DTranspose(1,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same',
                                            activation='sigmoid'))
        return


    def _build_self(self):
        # self.input = layers.input_layer(shape=(32,32))
        self.conv1 = layers.Conv2D(input_shape=(32,32), filters=self.filter_base_count,
                                   kernel_size=self.kernel_size,
                                   strides=self.strides,
                                   padding='same')
        self.conv2 = layers.Conv2D(2*self.filter_base_count,
                                   self.kernel_size,
                                   strides=self.strides,
                                   padding='same')
        self.conv3 = layers.Conv2D(4*self.filter_base_count,
                                   self.kernel_size,
                                   strides=self.strides,
                                   padding='same')
        self.encode_reshape = layers.Reshape((-1, 256*4*4*4))
        self.encode_dense = layers.Dense(self.code_dim, activation='relu')
 
        self.decode_dense = layers.Dense(256*4*4, activation='relu')
        self.decode_reshape = layers.Reshape((-1, 4, 4, 256))
        self.deconv1 = layers.Conv2DTranspose(4*self.filter_base_count,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same')
        self.deconv2 = layers.Conv2DTranspose(2*self.filter_base_count,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same')
        self.deconv3 = layers.Conv2DTranspose(1,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same',
                                            activation='sigmoid')
        return

   
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.encode_reshape(x)
        x = self.encode_dense(x)
        x = self.decode_dense(x)
        x = self.decode_reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class ImageEncoder(Sequential):

    def __init__(self, code_dim=128, input_dim=32, kernel_size=(5,5), filter_base_count=32,
                  strides=(2,2)):
        raise(Exception, "Now rolled into ae")
        super(ImageEncoder, self).__init__()
        self.code_dim = code_dim
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.filter_base_count = filter_base_count
        self.strides = strides
        self._build_self()
        return


    def _build_self(self):
        self.conv1 = layers.Conv2D(self.filter_base_count,
                                   self.kernel_size,
                                   strides=self.strides,
                                   padding='same')
        self.conv2 = layers.Conv2D(2*self.filter_base_count,
                                   self.kernel_size,
                                   strides=self.strides,
                                   padding='same')
        self.conv3 = layers.Conv2D(4*self.filter_base_count,
                                   self.kernel_size,
                                   strides=self.strides,
                                   padding='same')
        self.encode_reshape = layers.Reshape((-1, 256*4*4*4))
        self.encode_dense = layers.Dense(self.code_dim, activation='relu')

        self.decode_dense = layers.Dense(256*4*4, activation='relu')
        self.decode_reshape = layers.Reshape((-1, 4, 4, 256))
        self.deconv1 = layers.Conv2DTranspose(4*self.filter_base_count,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same')
        self.deconv2 = layers.Conv2DTranspose(2*self.filter_base_count,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same')
        self.deconv3 = layers.Conv2DTranspose(1,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same',
                                            activation='sigmoid')
        return


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.encode_reshape(x)
        x = self.encode_dense(x)
        x = self.decode_dense(x)
        x = self.decode_reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ImageDecoder(Sequential):
    
    def __init__(self, code_dim=128, output_dim=32, kernel_size=(5,5), filter_base_count=32,
                  strides=(2,2)):
        raise(Exception, "Now rolled into ae")
        super(ImageDecoder, self).__init__()
        self.code_dim = code_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.filter_base_count = filter_base_count
        self.strides = strides
        self._build_self()
        return
        

    def _build_self(self):
        self.decode_dense = layers.Dense(256*4*4, activation='relu')
        self.decode_reshape = layers.Reshape((-1, 4, 4, 256))
        self.deconv1 = layers.Conv2DTranspose(4*self.filter_base_count,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same')
        self.deconv2 = layers.Conv2DTranspose(2*self.filter_base_count,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same')
        self.deconv3 = layers.Conv2DTranspose(1,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same',
                                            activation='sigmoid')
        return


    def call(self, inputs):
        x = self.decode_dense(inputs)
        x = self.decode_reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x
"""
