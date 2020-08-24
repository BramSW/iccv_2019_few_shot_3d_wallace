from keras.models import Sequential, Model
from keras import layers, regularizers


def build_voxel_ae(code_dim=128, voxel_dim=32, kernel_size=(5,5,5), filter_base_count=32,
                  strides=(2,2,2), regularization=0.0, dropout_rate=0.0, use_batchnorm=False):
        model = Sequential()
        if voxel_dim == 128:
            model.add(layers.Conv3D(input_shape=(128,128,128,1), filters=4,
                                   kernel_size=kernel_size, strides=strides))
            model.add(layers.LeakyReLU())
            model.add(layers.Conv3D(filters=8,  kernel_size=kernel_size, strides=strides))
            model.add(layers.LeakyReLU())
            model.add(layers.Conv3D(filters=4,  kernel_size=(1,1,1), strides=(1,1,1)))
            model.add(layers.LeakyReLU())
            model.add(layers.Conv3D(filters=8,  kernel_size=kernel_size, strides=strides))
            model.add(layers.LeakyReLU())
            print(model.summary())
            model.add(layers.Reshape((-1,)))
            model.add(layers.Dense(512, name='vox_code', activation='relu'))
        elif voxel_dim == 32:
            # NEED TO REMOVE SOME OF THE HARDCODING HERE
            model.add(layers.Conv3D(input_shape=(32,32,32,1), filters=filter_base_count,
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       padding='same'))
            model.add(layers.LeakyReLU())
            if use_batchnorm: model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=dropout_rate))
            model.add(layers.Conv3D(2*filter_base_count,
                                       kernel_size,
                                       strides=strides,
                                       padding='same'))
            model.add(layers.LeakyReLU())
            if use_batchnorm: model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=dropout_rate))
            model.add(layers.Conv3D(4*filter_base_count,
                                       kernel_size,
                                       strides=strides,
                                       padding='same'))
            model.add(layers.LeakyReLU())
            if use_batchnorm: model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=dropout_rate))
            model.add(layers.Reshape((128*4*4*4,))) ##
            model.add(layers.Dense(512, activation='relu',
                                   kernel_regularizer=regularizers.l2(regularization)))
            if use_batchnorm: model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=dropout_rate))

            model.add(layers.Dense(code_dim, activation='relu', name='vox_code',
                                   kernel_regularizer=regularizers.l2(regularization)))
            if use_batchnorm: model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=dropout_rate))

            model.add(layers.Dense(512, activation='relu',
                                   kernel_regularizer=regularizers.l2(regularization)))
            if use_batchnorm: model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=dropout_rate))

            model.add(layers.Dense(128*4*4*4, activation='relu',  kernel_regularizer=regularizers.l2(regularization)))
            if use_batchnorm: model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=dropout_rate))
            model.add(layers.Reshape((4, 4, 4, 128)))
            model.add(layers.Conv3DTranspose(4*filter_base_count,
                                                kernel_size,
                                                strides=strides,
                                                padding='same',
                                                kernel_regularizer=regularizers.l2(regularization)))
            if use_batchnorm: model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=dropout_rate))
            model.add(layers.Conv3DTranspose(2*filter_base_count,
                                                kernel_size,
                                                strides=strides,
                                                padding='same',
                                                kernel_regularizer=regularizers.l2(regularization)))
            if use_batchnorm: model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=dropout_rate))
            model.add(layers.Conv3DTranspose(1,
                                                kernel_size,
                                                strides=strides,
                                                padding='same',
                                                activation='sigmoid', 
                                                kernel_regularizer=regularizers.l2(regularization)))
            if use_batchnorm: model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=dropout_rate))
        print(model.summary())
        return model


class VoxelAutoencoder():
    
    def __init__(self,code_dim=128, voxel_dim=32, kernel_size=(5,5,5), filter_base_count=32,
                  strides=(2,2,2)):
        super(VoxelAutoencoder, self).__init__()
        self.code_dim = code_dim
        self.voxel_dim = voxel_dim
        self.kernel_size = kernel_size
        self.filter_base_count = filter_base_count
        self.strides = strides
        self._build_self()
 
    def _build_self(self):
        self.model = Sequential()
        # NEED TO REMOVE SOME OF THE HARDCODING HERE
        self.model.add(layers.Conv3D(input_shape=(32,32,32,1), filters=self.filter_base_count,
                                   kernel_size=self.kernel_size,
                                   strides=self.strides,
                                   padding='same'))
        self.model.add(layers.Conv3D(2*self.filter_base_count,
                                   self.kernel_size,
                                   strides=self.strides,
                                   padding='same'))
        self.model.add(layers.Conv3D(4*self.filter_base_count,
                                   self.kernel_size,
                                   strides=self.strides,
                                   padding='same'))
        self.model.add(layers.Reshape((-1, 128*4*4*4)))
        self.model.add(layers.Dense(self.code_dim, activation='relu'))

        self.model.add(layers.Dense(128*4*4*4, activation='relu'))
        self.model.add(layers.Reshape((4, 4, 4, 128)))
        self.model.add(layers.Conv3DTranspose(4*self.filter_base_count,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same'))
        self.model.add(layers.Conv3DTranspose(2*self.filter_base_count,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same'))
        self.model.add(layers.Conv3DTranspose(1,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same',
                                            activation='sigmoid'))
        return

    def save_model(self, savepath):
        self.model.save(savepath)


"""
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
