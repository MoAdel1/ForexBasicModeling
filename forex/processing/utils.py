#%% code import
import math
import numpy as np
import talib as ta
from scipy.signal import argrelextrema


#%% main functions
def data_construct(DataFrame, lookUp, predictionWindow, pairName):
    '''function to construct the features from the inspection window and to create the supervised x,y pairs for training.

    Parameters
    ----------
    DataFrame : dataFrame
    LookUp : int
    predictionWindow : int
    pairName : str

    Returns
    -------
    output : dict
        a dict containing inputs matrix, targets matrix, raw inputs and mapping dict for features

    '''
    # fetch data for indicators calculations
    openPrice = DataFrame.o.values.astype("double")
    closePrice = DataFrame.c.values.astype("double")
    highPrice = DataFrame.h.values.astype("double")
    lowPrice = DataFrame.l.values.astype("double")
    volume = DataFrame.volume.values.astype("double")
    
    # calculate technical indicators values
    simple_ma_slow = ta.SMA(closePrice,30) # slow moving average 
    simple_ma_fast = ta.SMA(closePrice,15) # fast moving average 
    exp_ma_slow = ta.EMA(closePrice,20) # slow exp moving average 
    exp_ma_fast = ta.EMA(closePrice,10) # fast exp moving average 
    bbands = ta.BBANDS(closePrice,timeperiod=15) # calculate bollinger bands
    deltaBands = (bbands[0]-bbands[2])/bbands[2] # deltas between bands vector (bollinger)
    macd_s1,macd_s2,macd_hist = ta.MACD(closePrice) # MACD values calculation
    sar = ta.SAR(highPrice,lowPrice) # prabolic SAR
    stochK,stochD = ta.STOCH(highPrice,lowPrice,closePrice) # stochastic calculations
    rsi = ta.RSI(closePrice,timeperiod=15) # RSI indicator
    adx = ta.ADX(highPrice,lowPrice,closePrice,timeperiod=15) # ADX indicator
    mfi = ta.MFI(highPrice,lowPrice,closePrice,volume,timeperiod=15) # money flow index
    
    # calculate statistical indicators values
    beta = ta.BETA(highPrice,lowPrice,timeperiod=5) # beta from CAPM model
    slope = ta.LINEARREG_ANGLE(closePrice,timeperiod=5) # slope for fitting linera reg. to the last x points
    
    # calculate candle indicators values
    spinTop = ta.CDLSPINNINGTOP(openPrice,highPrice,lowPrice,closePrice)
    doji = ta.CDLDOJI(openPrice,highPrice,lowPrice,closePrice)
    dojiStar = ta.CDLDOJISTAR(openPrice,highPrice,lowPrice,closePrice)
    marubozu = ta.CDLMARUBOZU(openPrice,highPrice,lowPrice,closePrice)
    hammer = ta.CDLHAMMER(openPrice,highPrice,lowPrice,closePrice)
    invHammer = ta.CDLINVERTEDHAMMER(openPrice,highPrice,lowPrice,closePrice)
    hangingMan = ta.CDLHANGINGMAN(openPrice,highPrice,lowPrice,closePrice)
    shootingStar = ta.CDLSHOOTINGSTAR(openPrice,highPrice,lowPrice,closePrice)
    engulfing = ta.CDLENGULFING(openPrice,highPrice,lowPrice,closePrice)
    morningStar = ta.CDLMORNINGSTAR(openPrice,highPrice,lowPrice,closePrice)
    eveningStar = ta.CDLEVENINGSTAR(openPrice,highPrice,lowPrice,closePrice)
    whiteSoldier = ta.CDL3WHITESOLDIERS(openPrice,highPrice,lowPrice,closePrice)
    blackCrow = ta.CDL3BLACKCROWS(openPrice,highPrice,lowPrice,closePrice)
    insideThree = ta.CDL3INSIDE(openPrice,highPrice,lowPrice,closePrice)
    
    # prepare the final matrix
    '''
    matrix configurations ::> [o,c,h,l,ma_slow,ma_fast,exp_slow,exp_fast,
                           deltaBands,macd_s1,macd_s2,sar,stochK,
                           stochD,rsi,adx,mfi,beta,slope,spinTop,doji,dojiStar,
                           marubozu,hammer,invHammer,hangingMan,shootingStar,engulfing,
                           morningStar,eveningStar,whiteSoldier,blackCrow,insideThree]
    a 33 features matrix in total
    '''
    DataMatrix = np.column_stack((openPrice,closePrice,highPrice,lowPrice,
                              simple_ma_slow,simple_ma_fast,exp_ma_slow,
                              exp_ma_fast,deltaBands,macd_s1,macd_s2,
                              sar,stochK,stochD,rsi,adx,mfi,beta,slope,
                              spinTop,doji,dojiStar,marubozu,hammer,
                              invHammer,hangingMan,shootingStar,engulfing,
                              morningStar,eveningStar,whiteSoldier,blackCrow,
                              insideThree))
    
    # remove undifined values
    DataMatrix = DataMatrix[~np.isnan(DataMatrix).any(axis=1)] # remove all raws containing nan values
    
    # define number of windows to analyze
    framesCount = DataMatrix.shape[0]-(lookUp+predictionWindow)+1 # 1D convolution outputsize = ceil[((n-f)/s)+1]
    
    # define input/output arrays container
    rawInputs = {}
    inputsOpen = np.zeros((framesCount,lookUp))
    inputsClose = np.zeros((framesCount,lookUp))
    inputsHigh = np.zeros((framesCount,lookUp))
    inputsLow = np.zeros((framesCount,lookUp))
    inputs = np.zeros((framesCount,62))
    outputs = np.zeros((framesCount,1))
    
    # main loop and data
    for i in range(framesCount):
        mainFrame = DataMatrix[i:i+lookUp+predictionWindow,:] 
        window = np.array_split(mainFrame,[lookUp])[0]
        windowForecast = np.array_split(mainFrame,[lookUp])[1]
        '''
        window configurations ::>
        [0:o,1:c,2:h,3:l,4:ma_slow,5:ma_fast,6:exp_slow,7:exp_fast,
         8:deltaBands,9:macd_slow,10:macd_fast,11:sar,12:stochK,
         13:stochD,14:rsi,15:adx,16:mfi,17:beta,18:slope,19:spinTop,20:doji,21:dojiStar,
         22:marubozu,23:hammer,24:invHammer,25:hangingMan,26:shootingStar,27:engulfing,
         28:morningStar,29:eveningStar,30:whiteSoldier,31:blackCrow,32:insideThree]
        '''
        
        #sma features detection
        ma_slow = window[:,4]
        ma_fast = window[:,5]
        uptrend_cross = ma_fast>ma_slow
        uptrend_cross = np.concatenate((np.array([False]),(uptrend_cross[:-1]<uptrend_cross[1:]))) # check the false->true transition
        try:
            uptrend_cross_location = np.where(uptrend_cross==True)[0][-1] # latest uptrend cross_over location
        except:
            uptrend_cross_location = -1
        downtrend_cross = ma_slow>ma_fast
        downtrend_cross = np.concatenate((np.array([False]),(downtrend_cross[:-1]<downtrend_cross[1:]))) # check the false->true transition
        try:
            downtrend_cross_location = np.where(downtrend_cross==True)[0][-1] # latest downtrend cross_over location
        except:
            downtrend_cross_location = -1
        if(uptrend_cross_location > downtrend_cross_location): # latest cross is an uptrend
            sma_latest_crossover = 1 # uptrend sign
            sma_location_of_latest_crossover = uptrend_cross_location
            alpha_1 = (math.atan(ma_slow[uptrend_cross_location]-ma_slow[uptrend_cross_location-1]))*(180/math.pi)
            alpha_2 = (math.atan(ma_fast[uptrend_cross_location]-ma_fast[uptrend_cross_location-1]))*(180/math.pi)
            sma_latest_crossover_angle = alpha_1+alpha_2
        elif(downtrend_cross_location > uptrend_cross_location): # latest cross is a downtrend
            sma_latest_crossover = -1 # downtrend sign
            sma_location_of_latest_crossover = downtrend_cross_location
            alpha_1 = (math.atan(ma_slow[downtrend_cross_location]-ma_slow[downtrend_cross_location-1]))*(180/math.pi)
            alpha_2 = (math.atan(ma_fast[downtrend_cross_location]-ma_fast[downtrend_cross_location-1]))*(180/math.pi)
            sma_latest_crossover_angle = alpha_1+alpha_2
        else: # no cross in the given window
            sma_latest_crossover = 0 # no sign
            sma_location_of_latest_crossover = -1
            sma_latest_crossover_angle = 0
        up_count = np.sum(ma_fast>ma_slow)
        down_count = np.sum(ma_slow>ma_fast)
        if(up_count > down_count):
            sma_dominant_type_fast_slow = 1
        elif(down_count > up_count):
            sma_dominant_type_fast_slow = -1
        else:
            sma_dominant_type_fast_slow = 0

        #ema features detection
        exp_slow = window[:,6]
        exp_fast = window[:,7]
        uptrend_cross = exp_fast>exp_slow
        uptrend_cross = np.concatenate((np.array([False]),(uptrend_cross[:-1]<uptrend_cross[1:]))) # check the false->true transition
        try:
            uptrend_cross_location = np.where(uptrend_cross==True)[0][-1] # latest uptrend cross_over location
        except:
            uptrend_cross_location = -1
        downtrend_cross = exp_slow>exp_fast
        downtrend_cross = np.concatenate((np.array([False]),(downtrend_cross[:-1]<downtrend_cross[1:]))) # check the false->true transition
        try:
            downtrend_cross_location = np.where(downtrend_cross==True)[0][-1] # latest downtrend cross_over location
        except:
            downtrend_cross_location = -1
        if(uptrend_cross_location > downtrend_cross_location): # latest cross is an uptrend
            ema_latest_crossover = 1 # uptrend sign
            ema_location_of_latest_crossover = uptrend_cross_location
            alpha_1 = (math.atan(exp_slow[uptrend_cross_location]-exp_slow[uptrend_cross_location-1]))*(180/math.pi)
            alpha_2 = (math.atan(exp_fast[uptrend_cross_location]-exp_fast[uptrend_cross_location-1]))*(180/math.pi)
            ema_latest_crossover_angle = alpha_1+alpha_2
        elif(downtrend_cross_location > uptrend_cross_location): # latest cross is a downtrend
            ema_latest_crossover = -1 # downtrend sign
            ema_location_of_latest_crossover = downtrend_cross_location
            alpha_1 = (math.atan(exp_slow[downtrend_cross_location]-exp_slow[downtrend_cross_location-1]))*(180/math.pi)
            alpha_2 = (math.atan(exp_fast[downtrend_cross_location]-exp_fast[downtrend_cross_location-1]))*(180/math.pi)
            ema_latest_crossover_angle = alpha_1+alpha_2
        else: # no cross in the given window
            ema_latest_crossover = 0 # no sign
            ema_location_of_latest_crossover = -1
            ema_latest_crossover_angle = 0
        up_count = np.sum(exp_fast>exp_slow)
        down_count = np.sum(exp_slow>exp_fast)
        if(up_count > down_count):
            ema_dominant_type_fast_slow = 1
        elif(down_count > up_count):
            ema_dominant_type_fast_slow = -1
        else:
            ema_dominant_type_fast_slow = 0

        # B.Bands features detection
        deltaBands = window[:,8]
        deltaBands_mean = np.mean(deltaBands)
        deltaBands_std = np.std(deltaBands)
        deltaBands_maximum_mean = np.amax(deltaBands)/deltaBands_mean
        deltaBands_maximum_location = np.where(deltaBands==np.amax(deltaBands))[0][-1] # location of maximum
        deltaBands_minimum_mean = np.amin(deltaBands)/deltaBands_mean
        deltaBands_minimum_location = np.where(deltaBands==np.amin(deltaBands))[0][-1] # location of maximum

        # macd features detection
        macd_slow = window[:,9]
        macd_fast = window[:,10]
        uptrend_cross = macd_fast>macd_slow
        uptrend_cross = np.concatenate((np.array([False]),(uptrend_cross[:-1]<uptrend_cross[1:]))) # check the false->true transition
        try:
            uptrend_cross_location = np.where(uptrend_cross==True)[0][-1] # latest uptrend cross_over location
        except:
            uptrend_cross_location = -1
        downtrend_cross = macd_slow>macd_fast
        downtrend_cross = np.concatenate((np.array([False]),(downtrend_cross[:-1]<downtrend_cross[1:]))) # check the false->true transition
        try:
            downtrend_cross_location = np.where(downtrend_cross==True)[0][-1] # latest downtrend cross_over location
        except:
            downtrend_cross_location = -1
        if(uptrend_cross_location > downtrend_cross_location): # latest cross is an uptrend
            macd_latest_crossover = 1 # uptrend sign
            macd_location_of_latest_crossover = uptrend_cross_location
            alpha_1 = (math.atan(macd_slow[uptrend_cross_location]-macd_slow[uptrend_cross_location-1]))*(180/math.pi)
            alpha_2 = (math.atan(macd_fast[uptrend_cross_location]-macd_fast[uptrend_cross_location-1]))*(180/math.pi)
            macd_latest_crossover_angle = alpha_1+alpha_2
        elif(downtrend_cross_location > uptrend_cross_location): # latest cross is a downtrend
            macd_latest_crossover = -1 # downtrend sign
            macd_location_of_latest_crossover = downtrend_cross_location
            alpha_1 = (math.atan(macd_slow[downtrend_cross_location]-macd_slow[downtrend_cross_location-1]))*(180/math.pi)
            alpha_2 = (math.atan(macd_fast[downtrend_cross_location]-macd_fast[downtrend_cross_location-1]))*(180/math.pi)
            macd_latest_crossover_angle = alpha_1+alpha_2
        else: # no cross in the given window
            macd_latest_crossover = 0 # no sign
            macd_location_of_latest_crossover = -1
            macd_latest_crossover_angle = 0
        up_count = np.sum(macd_fast>macd_slow)
        down_count = np.sum(macd_slow>macd_fast)
        if(up_count > down_count):
            macd_dominant_type_fast_slow = 1
        elif(down_count > up_count):
            macd_dominant_type_fast_slow = -1
        else:
            macd_dominant_type_fast_slow = 0

        # sar features detection
        average_price = (window[:,0]+window[:,1]+window[:,2]+window[:,3])/4
        sar = window[:,11]
        uptrend = sar<average_price
        uptrend = np.concatenate((np.array([False]),(uptrend[:-1]<uptrend[1:]))) # check the false->true transition
        try:
            uptrend_location = np.where(uptrend==True)[0][-1] # latest uptrend location
        except:
            uptrend_location = -1
        downtrend = sar>average_price
        downtrend = np.concatenate((np.array([False]),(downtrend[:-1]<downtrend[1:]))) # check the false->true transition
        try:
            downtrend_location = np.where(downtrend==True)[0][-1] # latest downtrend location
        except:
            downtrend_location = -1
        if(uptrend_location > downtrend_location): # latest signal is an uptrend
            sar_latest_shiftPoint = 1
            sar_latest_shiftPoint_location = uptrend_location
        elif(downtrend_location > uptrend_location): # latest signal is a downtrend
            sar_latest_shiftPoint = -1
            sar_latest_shiftPoint_location = downtrend_location
        else: # same direction along the frame under question
            sar_latest_shiftPoint = 0 # no sign
            sar_latest_shiftPoint_location = -1
        sar_total_number_shifts = np.where(downtrend==True)[0].shape[0] + np.where(uptrend==True)[0].shape[0]

        # stochastic(K) features detection
        stochK = window[:,12]
        stochK_mean = np.mean(stochK)
        stochK_std = np.std(stochK)
        uptrend = stochK<=20
        uptrend = np.concatenate((np.array([False]),(uptrend[:-1]<uptrend[1:]))) # check the false->true transition
        try:
            uptrend_location = np.where(uptrend==True)[0][-1] # latest uptrend location
        except:
            uptrend_location = -1
        downtrend = stochK>=80 
        downtrend = np.concatenate((np.array([False]),(downtrend[:-1]<downtrend[1:]))) # check the false->true transition
        try:
            downtrend_location = np.where(downtrend==True)[0][-1] # latest downtrend location
        except:
            downtrend_location = -1
        if(uptrend_location > downtrend_location): # latest signal is an uptrend
            stochK_latest_event = 1
            stochK_event_location = uptrend_location
        elif(downtrend_location > uptrend_location): # latest signal is a downtrend
            stochK_latest_event = -1
            stochK_event_location = downtrend_location
        else: # same direction along the frame under question
            stochK_latest_event = 0 # no sign
            stochK_event_location = -1

        # stochastic(D) features detection
        stochD = window[:,13]
        stochD_mean = np.mean(stochD)
        stochD_std = np.std(stochD)
        uptrend = stochD<=20
        uptrend = np.concatenate((np.array([False]),(uptrend[:-1]<uptrend[1:]))) # check the false->true transition
        try:
            uptrend_location = np.where(uptrend==True)[0][-1] # latest uptrend location
        except:
            uptrend_location = -1
        downtrend = stochD>=80 
        downtrend = np.concatenate((np.array([False]),(downtrend[:-1]<downtrend[1:]))) # check the false->true transition
        try:
            downtrend_location = np.where(downtrend==True)[0][-1] # latest downtrend location
        except:
            downtrend_location = -1
        if(uptrend_location > downtrend_location): # latest signal is an uptrend
            stochD_latest_event = 1
            stochD_event_location = uptrend_location
        elif(downtrend_location > uptrend_location): # latest signal is a downtrend
            stochD_latest_event = -1
            stochD_event_location = downtrend_location
        else: # same direction along the frame under question
            stochD_latest_event = 0 # no sign
            stochD_event_location = -1

        # rsi features detection
        rsi = window[:,14]
        rsi_mean = np.mean(rsi)
        rsi_std = np.std(rsi)
        uptrend = rsi<=30
        uptrend = np.concatenate((np.array([False]),(uptrend[:-1]<uptrend[1:]))) # check the false->true transition
        try:
            uptrend_location = np.where(uptrend==True)[0][-1] # latest uptrend location
        except:
            uptrend_location = -1
        downtrend = rsi>=70 
        downtrend = np.concatenate((np.array([False]),(downtrend[:-1]<downtrend[1:]))) # check the false->true transition
        try:
            downtrend_location = np.where(downtrend==True)[0][-1] # latest downtrend location
        except:
            downtrend_location = -1
        if(uptrend_location > downtrend_location): # latest signal is an uptrend
            rsi_latest_event = 1
            rsi_event_location = uptrend_location
        elif(downtrend_location > uptrend_location): # latest signal is a downtrend
            rsi_latest_event = -1
            rsi_event_location = downtrend_location
        else: # same direction along the frame under question
            rsi_latest_event = 0 # no sign
            rsi_event_location = -1

        # adx features detection
        adx = window[:,15]
        adx_mean = np.mean(adx)
        adx_std = np.std(adx)
        splitted_array = np.array_split(adx,2)
        m0 = np.mean(splitted_array[0])
        m1 = np.mean(splitted_array[1])
        adx_mean_delta_bet_first_second_half = (m1-m0)/m0

        # mfi features detection
        mfi = window[:,16]
        mfi_mean = np.mean(mfi)
        mfi_std = np.std(mfi)
        splitted_array = np.array_split(mfi,2)
        m0 = np.mean(splitted_array[0])
        m1 = np.mean(splitted_array[1])
        mfi_mean_delta_bet_first_second_half = (m1-m0)/m0

        # resistance levels features detection
        closePrice = window[:,1]
        resLevels = argrelextrema(closePrice,np.greater,order=4)[0]
        if(resLevels.shape[0]==0):
            relation_r1_close = 0
            relation_r2_close = 0
            relation_r3_close = 0
        elif(resLevels.shape[0]==1):
            relation_r1_close = (closePrice[-1]-closePrice[resLevels[-1]])/closePrice[-1]
            relation_r2_close = 0
            relation_r3_close = 0
        elif(resLevels.shape[0]==2):
            relation_r1_close = (closePrice[-1]-closePrice[resLevels[-1]])/closePrice[-1]
            relation_r2_close = (closePrice[-1]-closePrice[resLevels[-2]])/closePrice[-1]
            relation_r3_close = 0
        else:
            relation_r1_close = (closePrice[-1]-closePrice[resLevels[-1]])/closePrice[-1]
            relation_r2_close = (closePrice[-1]-closePrice[resLevels[-2]])/closePrice[-1]
            relation_r3_close = (closePrice[-1]-closePrice[resLevels[-3]])/closePrice[-1]

        # support levels features detection
        closePrice = window[:,1]
        supLevels = argrelextrema(closePrice,np.less,order=4)[0]
        if(supLevels.shape[0]==0):
            relation_s1_close = 0
            relation_s2_close = 0
            relation_s3_close = 0
        elif(supLevels.shape[0]==1):
            relation_s1_close = (closePrice[-1]-closePrice[supLevels[-1]])/closePrice[-1]
            relation_s2_close = 0
            relation_s3_close = 0
        elif(supLevels.shape[0]==2):
            relation_s1_close = (closePrice[-1]-closePrice[supLevels[-1]])/closePrice[-1]
            relation_s2_close = (closePrice[-1]-closePrice[supLevels[-2]])/closePrice[-1]
            relation_s3_close = 0
        else:
            relation_s1_close = (closePrice[-1]-closePrice[supLevels[-1]])/closePrice[-1]
            relation_s2_close = (closePrice[-1]-closePrice[supLevels[-2]])/closePrice[-1]
            relation_s3_close = (closePrice[-1]-closePrice[supLevels[-3]])/closePrice[-1]

        # slope features detection
        slope = window[:,18]
        slope_mean = np.mean(slope)

        # beta features detection
        beta = window[:,17]
        beta_mean = np.mean(beta)
        beta_std = np.std(beta)

        # spinTop features detection    np.sum(np.where(a==1)[0])
        count100plus = np.sum(np.where(window[:,19]==100)[0])
        count100minus = (np.sum(np.where(window[:,19]==-100)[0]))*-1
        spinTop_number_occurrence = count100plus+count100minus

        # doji features detection
        count100plus = np.sum(np.where(window[:,20]==100)[0])
        count100minus = (np.sum(np.where(window[:,20]==-100)[0]))*-1
        doji_number_occurrence = count100plus+count100minus

        # dojiStar features detection
        count100plus = np.sum(np.where(window[:,21]==100)[0])
        count100minus = (np.sum(np.where(window[:,21]==-100)[0]))*-1
        dojiStar_number_occurrence = count100plus+count100minus

        # marubozu features detection
        count100plus = np.sum(np.where(window[:,22]==100)[0])
        count100minus = (np.sum(np.where(window[:,22]==-100)[0]))*-1
        marubozu_number_occurrence = count100plus+count100minus

        # hammer features detection
        count100plus = np.sum(np.where(window[:,23]==100)[0])
        count100minus = (np.sum(np.where(window[:,23]==-100)[0]))*-1
        hammer_number_occurrence = count100plus+count100minus

        # invHammer features detection
        count100plus = np.sum(np.where(window[:,24]==100)[0])
        count100minus = (np.sum(np.where(window[:,24]==-100)[0]))*-1
        invHammer_number_occurrence = count100plus+count100minus

        # hangingMan features detection
        count100plus = np.sum(np.where(window[:,25]==100)[0])
        count100minus = (np.sum(np.where(window[:,25]==-100)[0]))*-1
        hangingMan_number_occurrence = count100plus+count100minus

        # shootingStar features detection
        count100plus = np.sum(np.where(window[:,26]==100)[0])
        count100minus = (np.sum(np.where(window[:,26]==-100)[0]))*-1
        shootingStar_number_occurrence = count100plus+count100minus

        # engulfing features detection
        count100plus = np.sum(np.where(window[:,27]==100)[0])
        count100minus = (np.sum(np.where(window[:,27]==-100)[0]))*-1
        engulfing_number_occurrence = count100plus+count100minus

        # morningStar features detection
        count100plus = np.sum(np.where(window[:,28]==100)[0])
        count100minus = (np.sum(np.where(window[:,28]==-100)[0]))*-1
        morningStar_number_occurrence = count100plus+count100minus

        # eveningStar features detection
        count100plus = np.sum(np.where(window[:,29]==100)[0])
        count100minus = (np.sum(np.where(window[:,29]==-100)[0]))*-1
        eveningStar_number_occurrence = count100plus+count100minus

        # whiteSoldier features detection
        count100plus = np.sum(np.where(window[:,30]==100)[0])
        count100minus = (np.sum(np.where(window[:,30]==-100)[0]))*-1
        whiteSoldier_number_occurrence = count100plus+count100minus

        # blackCrow features detection
        count100plus = np.sum(np.where(window[:,31]==100)[0])
        count100minus = (np.sum(np.where(window[:,31]==-100)[0]))*-1
        blackCrow_number_occurrence = count100plus+count100minus

        # insideThree features detection
        count100plus = np.sum(np.where(window[:,32]==100)[0])
        count100minus = (np.sum(np.where(window[:,32]==-100)[0]))*-1
        insideThree_number_occurrence = count100plus+count100minus

        # fill the inputs matrix
        inputs[i,0] = sma_latest_crossover
        inputs[i,1] = sma_location_of_latest_crossover
        inputs[i,2] = sma_latest_crossover_angle
        inputs[i,3] = sma_dominant_type_fast_slow
        inputs[i,4] = ema_latest_crossover
        inputs[i,5] = ema_location_of_latest_crossover
        inputs[i,6] = ema_latest_crossover_angle
        inputs[i,7] = ema_dominant_type_fast_slow
        inputs[i,8] = deltaBands_mean
        inputs[i,9] = deltaBands_std
        inputs[i,10] = deltaBands_maximum_mean
        inputs[i,11] = deltaBands_maximum_location
        inputs[i,12] = deltaBands_minimum_mean
        inputs[i,13] = deltaBands_minimum_location
        inputs[i,14] = macd_latest_crossover
        inputs[i,15] = macd_location_of_latest_crossover
        inputs[i,16] = macd_latest_crossover_angle
        inputs[i,17] = macd_dominant_type_fast_slow
        inputs[i,18] = sar_latest_shiftPoint
        inputs[i,19] = sar_latest_shiftPoint_location
        inputs[i,20] = sar_total_number_shifts
        inputs[i,21] = stochK_mean
        inputs[i,22] = stochK_std
        inputs[i,23] = stochK_latest_event
        inputs[i,24] = stochK_event_location
        inputs[i,25] = stochD_mean
        inputs[i,26] = stochD_std
        inputs[i,27] = stochD_latest_event
        inputs[i,28] = stochD_event_location
        inputs[i,29] = rsi_mean
        inputs[i,30] = rsi_std
        inputs[i,31] = rsi_latest_event
        inputs[i,32] = rsi_event_location
        inputs[i,33] = adx_mean
        inputs[i,34] = adx_std
        inputs[i,35] = adx_mean_delta_bet_first_second_half
        inputs[i,36] = mfi_mean
        inputs[i,37] = mfi_std
        inputs[i,38] = mfi_mean_delta_bet_first_second_half
        inputs[i,39] = relation_r1_close
        inputs[i,40] = relation_r2_close
        inputs[i,41] = relation_r3_close
        inputs[i,42] = relation_s1_close
        inputs[i,43] = relation_s2_close
        inputs[i,44] = relation_s3_close
        inputs[i,45] = slope_mean
        inputs[i,46] = beta_mean
        inputs[i,47] = beta_std
        inputs[i,48] = spinTop_number_occurrence
        inputs[i,49] = doji_number_occurrence
        inputs[i,50] = dojiStar_number_occurrence 
        inputs[i,51] = marubozu_number_occurrence
        inputs[i,52] = hammer_number_occurrence
        inputs[i,53] = invHammer_number_occurrence
        inputs[i,54] = hangingMan_number_occurrence
        inputs[i,55] = shootingStar_number_occurrence
        inputs[i,56] = engulfing_number_occurrence
        inputs[i,57] = morningStar_number_occurrence
        inputs[i,58] = eveningStar_number_occurrence
        inputs[i,59] = whiteSoldier_number_occurrence
        inputs[i,60] = blackCrow_number_occurrence
        inputs[i,61] = insideThree_number_occurrence

        # fill raw inputs matrices
        inputsOpen[i,:] = window[:,0].reshape(1,lookUp)
        inputsClose[i,:] = window[:,1].reshape(1,lookUp)
        inputsHigh[i,:] = window[:,2].reshape(1,lookUp)
        inputsLow[i,:] = window[:,3].reshape(1,lookUp)
        

        # fill the output matrix
        futureClose = windowForecast[:,1]
        if(pairName=="USD_JPY"):
            outputs[i,0] = (futureClose[-1]-futureClose[0])/0.01   # one pip = 0.01 for any pair containing JPY
        else:
            outputs[i,0] = (futureClose[-1]-futureClose[0])/0.0001 # one pip = 0.0001 for this pairs


    # create mapping dict.
    mappingDict = {
                    "sma_latest_crossover":0,
                    "sma_location_of_latest_crossover":1,
                    "sma_latest_crossover_angle":2,
                    "sma_dominant_type_fast_slow":3,
                    "ema_latest_crossover":4,
                    "ema_location_of_latest_crossover":5,
                    "ema_latest_crossover_angle":6,
                    "ema_dominant_type_fast_slow":7,
                    "deltaBands_mean":8,
                    "deltaBands_std":9,
                    "deltaBands_maximum_mean":10,
                    "deltaBands_maximum_location":11,
                    "deltaBands_minimum_mean":12,
                    "deltaBands_minimum_location":13,
                    "macd_latest_crossover":14,
                    "macd_location_of_latest_crossover":15,
                    "macd_latest_crossover_angle":16,
                    "macd_dominant_type_fast_slow":17,
                    "sar_latest_shiftPoint":18,
                    "sar_latest_shiftPoint_location":19,
                    "sar_total_number_shifts":20,
                    "stochK_mean":21,
                    "stochK_std":22,
                    "stochK_latest_event":23,
                    "stochK_event_location":24,
                    "stochD_mean":25,
                    "stochD_std":26,
                    "stochD_latest_event":27,
                    "stochD_event_location":28,
                    "rsi_mean":29,
                    "rsi_std":30,
                    "rsi_latest_event":31,
                    "rsi_event_location":32,
                    "adx_mean":33,
                    "adx_std":34,
                    "adx_mean_delta_bet_first_second_half":35,
                    "mfi_mean":36,
                    "mfi_std":37,
                    "mfi_mean_delta_bet_first_second_half":38,
                    "relation_r1_close":39,
                    "relation_r2_close":40,
                    "relation_r3_close":41,
                    "relation_s1_close":42,
                    "relation_s2_close":43,
                    "relation_s3_close":44,
                    "slope_mean":45,
                    "beta_mean":46,
                    "beta_std":47,
                    "spinTop_number_occurrence":48,
                    "doji_number_occurrence":49,
                    "dojiStar_number_occurrence":50,
                    "marubozu_number_occurrence":51,
                    "hammer_number_occurrence":52,
                    "invHammer_number_occurrence":53,
                    "hangingMan_number_occurrence":54,
                    "shootingStar_number_occurrence":55,
                    "engulfing_number_occurrence":56,
                    "morningStar_number_occurrence":57,
                    "eveningStar_number_occurrence":58,
                    "whiteSoldier_number_occurrence":59,
                    "blackCrow_number_occurrence":60,
                    "insideThree_number_occurrence":61}

    # remove undifined values from the output
    refMatrix = inputs
    inputs = inputs[~np.isnan(refMatrix).any(axis=1)] # remove all raws containing nan values
    outputs = outputs[~np.isnan(refMatrix).any(axis=1)] # remove all raws containing nan values
    inputsOpen = inputsOpen[~np.isnan(refMatrix).any(axis=1)] # remove all raws containing nan values
    inputsClose = inputsClose[~np.isnan(refMatrix).any(axis=1)] # remove all raws containing nan values
    inputsHigh = inputsHigh[~np.isnan(refMatrix).any(axis=1)] # remove all raws containing nan values
    inputsLow = inputsLow[~np.isnan(refMatrix).any(axis=1)] # remove all raws containing nan values

    # create raw inputs dict.
    rawInputs["open"] = inputsOpen
    rawInputs["close"] = inputsClose
    rawInputs["high"] = inputsHigh
    rawInputs["low"] = inputsLow

    # return the function output
    output = {"mappingDict":mappingDict,"rawInputs":rawInputs,"inputFeatures":inputs,"targets":outputs}
    return(output)


