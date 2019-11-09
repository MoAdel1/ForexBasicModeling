#%% code imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#%% functions definition
def split_data(x, y, val_model, val_optimizer):
    '''function to split the data into train and validation sets for ML models and optmizer

    Parameters
    ----------
    x : numpy ndarray
    y : numpy ndarray
    val_model : float
        percentage of data to be hold for validation of models
    val_optimizer : float
        percentage of data to be hold for optimizer testing
    
    Returns
    -------
    x_train : numpy ndarray
    y_train : numpy ndarray
    x_model_val : numpy ndarray
    y_model_val : numpy ndarray
    x_optimizer_val : numpy ndarray
    y_optimizer_val : numpy ndarray
    '''
    assert (val_model+val_optimizer) <= 0.4, "Validation datasets can not be more than 40%"
    size_model_val = int(x.shape[0]*val_model)
    size_optimizer_val = int(x.shape[0]*val_optimizer)
    # start splitting the data
    x_temp, x_model_val, y_temp, y_model_val = train_test_split(x, 
                                                                y, 
                                                                test_size = size_model_val)
    x_train, x_optimizer_val, y_train, y_optimizer_val = train_test_split(x_temp, 
                                                                          y_temp, 
                                                                          test_size = size_optimizer_val)
    return x_train, y_train, x_model_val, y_model_val, x_optimizer_val, y_optimizer_val

def standardize(x):
    '''function to standardize training data to have zero mean and std of one

    Parameters
    ----------
    x : numpy ndarray 

    Returns
    -------
    x_new : numpy ndarray
    scaler : StandardScaler object
    '''
    scaler = StandardScaler()
    scaler.fit(x)
    x_new = scaler.transform(x)
    return x_new, scaler
