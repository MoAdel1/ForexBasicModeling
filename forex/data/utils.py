#%% code import
import oandapyV20
import pandas as pd
from forex.configs.load_configs import configs
import oandapyV20.endpoints.instruments as instruments


#%% main functions
def fetch_data(granularity, count, instrument, return_list=False):
    '''function to fetch the raw data for the given instrument

    Parameters
    ----------
    granularity : str
        the granularity of the data fetched
    count : int
        the number of candle sticks to get from latest date, maximum (5000)
    instrument : str
        the currency pair to download
    return_list : bool
        the type of the return value

    Returns
    -------
    candles_data : dataFrame
        a pandas dataframe for (o, h, l, c) and other relative data
    '''
    # define call params
    params = {"granularity":granularity,
              "price":"M",
              "count":count}
    # create the client and make the call
    client = oandapyV20.API(access_token=configs['token'])
    data = list()
    r = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(r)
    data = r.response["candles"]
    # formate response into a data frame
    if(r.EXPECTED_STATUS==r.status_code and  len(data)!=0 and not(return_list)):
        required_data = pd.DataFrame(data)
        time = list(required_data["time"])
        volume = list(required_data["volume"])
        completed = list(required_data["complete"])
        prices_df = pd.DataFrame(list(required_data["mid"]))
        relevant_info_df = pd.DataFrame({"time":time,"completed":completed,"volume":volume})
        candles_data = pd.concat([relevant_info_df,prices_df],axis=1)
        return(candles_data)
    elif(r.EXPECTED_STATUS==r.status_code and  len(data)!=0 and return_list):
        return(data)
    else:
        return(-1)

