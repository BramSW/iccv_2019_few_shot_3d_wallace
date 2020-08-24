from keras.models import Sequential, Model
from keras import layers, regularizers


class VoxelRefiner():
    
    def __init__(self,image_ae, voxel_ae, 
                  voxel_dim=32, image_dim=32,
                  kernel_size=(5,5,5), filter_base_count=32,
                  strides=(2,2,2), regularization=0, dropout_rate=0.0,
                  use_batchnorm=False, concat_features=False, delta=True, output_dim=32, num_im_channels=1,
                  special_vox_shape=None, final_sigmoid=False,
                  tanh_and_add=True):
        super(VoxelRefiner, self).__init__()
        assert (not (tanh_and_add and final_sigmoid))
        self.delta = delta
        self.image_dim = image_dim
        self.voxel_dim = voxel_dim
        self.kernel_size = kernel_size
        self.filter_base_count = filter_base_count
        self.strides = strides
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.concat_features = concat_features
        self.num_im_channels = num_im_channels
        self.image_encoder = Model(inputs=image_ae.input, output=image_ae.get_layer('im_code').output)
        self.voxel_encoder = Model(inputs=voxel_ae.input, output=voxel_ae.get_layer('vox_code').output)
        self.output_dim = output_dim
        self.special_vox_shape = special_vox_shape
        self.final_sigmoid = final_sigmoid
        self.tanh_and_add = tanh_and_add
        self._build_self()
 
    def _build_self(self):
        # NEED TO REMOVE SOME OF THE HARDCODING HERE
        # Look into freezing encoders
        im_input = layers.Input(shape=(self.image_dim, self.image_dim, self.num_im_channels))
        vox_shape = (self.voxel_dim, self.voxel_dim, self.voxel_dim,1) if not self.special_vox_shape else self.special_vox_shape
        vox_input = layers.Input(shape=vox_shape)
        im_code = self.image_encoder(im_input)
        vox_code = self.voxel_encoder(vox_input)
        if self.concat_features: total_code = layers.concatenate([im_code, vox_code], axis=-1)
        else: total_code = layers.add([im_code, vox_code])
        
        if self.output_dim == 32:
            x = layers.Dense(128*4*4*4, activation='relu', 
                             kernel_regularizer=regularizers.l2(self.regularization))(total_code)
            x = layers.Dropout(rate=self.dropout_rate)(x)
            if self.use_batchnorm: x = layers.BatchNormalization()(x)
            x = layers.Reshape((4, 4, 4, 128))(x)
            x = layers.Conv3DTranspose(4*self.filter_base_count,
                                                self.kernel_size,
                                                strides=self.strides,
                                                padding='same',  
                                                kernel_regularizer=regularizers.l2(self.regularization),
                                                activation='relu')(x)
            if self.use_batchnorm: x = layers.BatchNormalization()(x)
            x = layers.Dropout(rate=self.dropout_rate)(x)
            x = layers.Conv3DTranspose(2*self.filter_base_count,
                                                self.kernel_size,
                                                strides=self.strides,
                                                padding='same', 
                                                kernel_regularizer=regularizers.l2(self.regularization),
                                                activation='relu')(x)
            if self.use_batchnorm: x = layers.BatchNormalization()(x)
            x = layers.Dropout(rate=self.dropout_rate)(x)
            x = layers.Conv3DTranspose(1,
                                                self.kernel_size,
                                                strides=self.strides,
                                                padding='same',
                                                activation='linear', 
                                                kernel_regularizer=regularizers.l2(self.regularization))(x)
            if self.use_batchnorm: x = layers.BatchNormalization()(x)
                
                
                        
            if self.tanh_and_add:
                print("USING TANH AND ADD")
                x = layers.Activation('tanh')(x)
                # x = layers.Add()([x, vox_input])
            else:
                if self.delta: x = layers.Add()([x, vox_input]) 
                    
                if self.final_sigmoid: x = layers.Activation('sigmoid')(x)
                else: x = layers.Activation('relu')(x)
                 
            self.refiner = Model(inputs=[im_input, vox_input], outputs=x)
        elif self.output_dim == 128:
            x = layers.Dense(4*4*4*128, activation='relu')(total_code)
            x = layers.Reshape((4,4,4,128))(x)
            x = layers.Conv3DTranspose(kernel_size=(6,6,6), strides=(2,2,2), filters=32)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Conv3DTranspose(kernel_size=(6,6,6), strides=(2,2,2), filters=16)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Conv3DTranspose(kernel_size=(6,6,6), strides=(2,2,2), filters=4)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Conv3DTranspose(kernel_size=(10,10,10), strides=(2,2,2), filters=1)(x)
            
            if self.tanh_and_add:
                x = layers.Activation('tanh')(x)
                x = layers.Add()([x, vox_input])
            else:
                if self.delta: x = layers.Add()([x, vox_input]) 
                    
                if self.final_sigmoid: x = layers.Activation('sigmoid')(x)
                else: x = layers.Activation('relu')(x)
                    
            self.refiner = Model(inputs=[im_input, vox_input], outputs=x)
            print(self.refiner.summary())
        return

