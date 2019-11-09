#%% code imports
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


#%% functions definition
def build_model(x_train, y_train, x_model_val, y_model_val, configs, problem_type):
    '''function to build the feed forward neural model

    Parameters
    ----------
    x_train : numpy ndarray
    y_train : numpy ndarray
    x_model_val : numpy ndarray
    y_model_val : numpy ndarray
    configs : list of booleans
    problem_type : str

    Returns
    -------
    model : TF model
    '''
    assert problem_type in ('classification', 'regression'), "Problem_type must be: classification or regression"
    number_layers = int('{}{}'.format(configs[0], configs[1]), 2) + 1
    size_layer = int('{}{}{}'.format(configs[2], configs[3], configs[4]), 2) + 1
    output_activation = 'linear' if problem_type=='regression' else 'sigmoid'
    loss = 'mean_squared_error' if problem_type=='regression' else 'binary_crossentropy'
    # build the model
    model = Sequential()
    for i in range(number_layers):
        if(i==0):
            model.add(Dense(size_layer, 
                            activation='relu', 
                            kernel_regularizer=l2(0.01),
                            input_dim=int(x_train.shape[1])))
        else:
            model.add(Dense(size_layer, 
                            activation='relu', 
                            kernel_regularizer=l2(0.01)))
    model.add(Dense(int(y_train.shape[1]), activation=output_activation))
    # set the optimizer and loss
    model.compile(optimizer='Adam', loss=loss)
    # specify the callback for early stopping and selecting the best model
    call_back = EarlyStopping(monitor='val_loss',
                              min_delta=0.0000000000001, 
                              patience=10, 
                              mode='min',
                              restore_best_weights=True)
    # start the training process
    model.fit(x_train, y_train, epochs=100,
              batch_size=32, shuffle=True,
              validation_data=(x_model_val, y_model_val),
              callbacks=[call_back],
              verbose=0)
    # function return
    return(model)
