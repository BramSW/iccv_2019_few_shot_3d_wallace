from keras.models import Sequential, Model
from keras import layers, regularizers


class VoxelRefiner():
    
    def __init__(self,image_ae, voxel_ae, 
                  voxel_dim=32, image_dim=32,
                  kernel_size=(5,5,5), filter_base_count=32,
                  strides=(2,2,2), regularization=0, dropout_rate=0.0):
        super(VoxelRefiner, self).__init__()
        self.image_dim = image_dim
        self.voxel_dim = voxel_dim
        self.kernel_size = kernel_size
        self.filter_base_count = filter_base_count
        self.strides = strides
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        self.image_encoder = Model(inputs=image_ae.input, output=image_ae.get_layer('im_code').output)
        self.voxel_encoder = Model(inputs=voxel_ae.input, output=voxel_ae.get_layer('vox_code').output)
        self._build_self()
 
    def _build_self(self):
        # NEED TO REMOVE SOME OF THE HARDCODING HERE
        # Look into freezing encoders
        im_input = layers.Input(shape=(self.image_dim, self.image_dim, 1))
        vox_input = layers.Input(shape=(self.voxel_dim, self.voxel_dim, self.voxel_dim,1))
        im_code = self.image_encoder(im_input)
        vox_code = self.voxel_encoder(vox_input)
        total_code = layers.concatenate([im_code, vox_code], axis=-1)
        x = layers.Dense(128*4*4*4, activation='relu', 
                         kernel_regularizer=regularizers.l2(self.regularization))(total_code)
        x = layers.Dropout(rate=self.dropout_rate)(x)
        x = layers.Reshape((4, 4, 4, 128))(x)
        x = layers.Conv3DTranspose(4*self.filter_base_count,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same',  
                                            kernel_regularizer=regularizers.l2(self.regularization))(x)
        x = layers.Dropout(rate=self.dropout_rate)(x)
        x = layers.Conv3DTranspose(2*self.filter_base_count,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same', 
                                            kernel_regularizer=regularizers.l2(self.regularization))(x)
        x = layers.Dropout(rate=self.dropout_rate)(x)
        x = layers.Conv3DTranspose(1,
                                            self.kernel_size,
                                            strides=self.strides,
                                            padding='same',
                                            activation='sigmoid', 
                                            kernel_regularizer=regularizers.l2(self.regularization))(x)

        self.refiner = Model(inputs=[im_input, vox_input], outputs=x)
        return

