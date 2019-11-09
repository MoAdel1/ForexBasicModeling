#%% code imports
import pandas as pd


#%% functions definition
def load_data(location, output_name):
    '''function to load data into a pandas data frame

    Parameters
    ----------
    location : str
        the location of the tabular data to be loaded
    output_name : str
        the name of the output column

    Returns
    -------
    x: numpy ndarray
        the data training feature
    y: numpy ndarray
        the data taget column
    '''
    df = pd.read_excel(location,
                       sheet_name=0,
                       header=0)
    features = list(df.columns)
    features.remove(output_name)
    x = df[features].values
    y = df[output_name].values
    return x, y
