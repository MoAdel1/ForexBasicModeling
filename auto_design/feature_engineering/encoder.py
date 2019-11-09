#%% code imports
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


#%% functions definition
def build_encoder(x_train, x_model_val, configs):
    '''function to build the encoder network for feature engineering stage

    Parameters
    ----------
    x_train : numpy ndarray
    x_model_val : numpy ndarray
    configs : list of booleans

    Returns
    -------
    encoder : TF model
    '''
    use_encoder = bool(configs[0])
    factor = 0.5 if bool(configs[1]) else 0.75
    if(use_encoder):
        # build the network
        input_layer = Input(shape=(int(x_train.shape[1]),))
        encoded = Dense(int(x_train.shape[1]*factor), activation='relu')(input_layer)
        encoded = Dense(int(x_train.shape[1]*factor**2), activation='relu')(encoded)
        encoded = Dense(int(x_train.shape[1]*factor**3), activation='relu')(encoded)
        decoded = Dense(int(x_train.shape[1]*factor**2), activation='relu')(encoded)
        decoded = Dense(int(x_train.shape[1]*factor), activation='relu')(decoded)
        decoded = Dense(int(x_train.shape[1]), activation='linear')(decoded)
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)
        # set the optimizer and loss
        autoencoder.compile(optimizer='Adam', loss='mean_squared_error')
        # specify the callback for early stopping and selecting the best model
        call_back = EarlyStopping(monitor='val_loss',
                                  min_delta=0.0000000000001, 
                                  patience=10, 
                                  mode='min',
                                  restore_best_weights=True)
        # start the training process
        autoencoder.fit(x_train, x_train, epochs=100,
                        batch_size=32, shuffle=True,
                        validation_data=(x_model_val, x_model_val),
                        callbacks=[call_back],
                        verbose=0)
        # function return
        return(encoder)
    else:
        return(None)

