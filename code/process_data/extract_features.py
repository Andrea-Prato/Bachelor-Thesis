
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import periodogram
import GSR_analysis as GSR
import HRV_analysis as HRV
import numpy as np
import os
from scipy.stats import pearsonr
import seaborn as sns
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

#initialize feature names
stat_features_names_all = ['mean','std','skew','kurtosis','diff','diff2','q25','q75',
                           'qdev','max-min','coeff_var','d_mean','d_std','d_skew','d_kurtosis',
                           'd_diff','d_diff2','d_q25','d_q75','d_qdev','d_max-min','d_coeff_var']

acc_features = ['acc_magnitude','acc_f1','acc_f2','acc_f3',
                'acc_f1_value','acc_f2_value','acc_f3_value']

feature_names = []
# bvp_f,st_f,eda_f,acc_f,hrv_feats,hrv_feats_ibi,corr_f

#generate a list with all features
for n in stat_features_names_all:
    feature_names.append('bvp_stat_'+n) 

for n in stat_features_names_all:
    feature_names.append('st_'+n)

for n in stat_features_names_all:
    feature_names.append('eda_stat_'+n)           

for n in GSR.feature_names_all:
    feature_names.append('eda_'+n)

for n in acc_features:
    feature_names.append(n)

for n in HRV.feature_names_time:
    feature_names.append('HRV_BVP_'+n)    

for n in HRV.feature_names_time:
    feature_names.append('HRV_IBI_'+n)    

feature_names.append('eda_st_corr')

for n in stat_features_names_all:
    feature_names.append('hr_'+n)
    
    #feature extraction functions
def get_stat_features_pd(arr):
    r = [arr.mean(),arr.std(),arr.skew(),arr.kurtosis(),arr.diff().mean(),
         arr.diff().diff().mean(),arr.quantile(0.25),arr.quantile(0.75),
         arr.quantile(0.75)-arr.quantile(0.25),arr.max()-arr.min()]
    if(arr.std()!=0):
        r.append(arr.mean()/arr.std())
    else:
         r.append(0.0)
    return np.hstack(r)

def get_stat_features_pd_all(arr):
    r = get_stat_features_pd(arr) #calculate statistical featues for the raw signal
    diff = arr.diff()
    diff.iloc[0] = diff.iloc[1]
    d = get_stat_features_pd(arr) #calculate statistical featues for the first derivative
        
    return np.hstack([r,d])

def extract_acc_features(signal):
    acc_freq = 32
    magnitude = np.sqrt(signal.x*signal.x + signal.y*signal.y+signal.x*signal.x)
    
    f, Pxx_den = periodogram(magnitude, acc_freq)
    top_3_idx = Pxx_den.argsort()[-3:][::-1]
    top_3_freqs = (f[top_3_idx]) #array of top 3 frequencies
    top_3_freqs_values = (Pxx_den[top_3_idx]) #array of magnitudes of top 3 frequencies
    
    result = np.hstack([np.mean(magnitude),top_3_freqs,top_3_freqs_values])
    return result

def extract_bvp_features(signal):
    f = get_stat_features_pd_all(pd.Series(signal))
    return f

def extract_st_features(signal):
    f = get_stat_features_pd_all(pd.Series(signal))
    return f

def extract_eda_features(signal,sampling_rate=4,plt_flag=False):
    f_sta = get_stat_features_pd_all(pd.Series(signal))
    if plt_flag:
        plt.figure(figsize=(30,3))
        plt.step(np.arange(len(signal)),signal,label='Original')
        moving_average_win_size = 2*sampling_rate #2 seconds
        signal = signal.rolling(moving_average_win_size).mean().iloc[moving_average_win_size:]
        plt.step(np.arange(len(signal)),signal,label='Original')
        # plt.show()
    f = GSR.get_GSR_features(signal.values, sampling_rate, height_threshold=.25, plt_flag=plt_flag)
    return np.hstack([f_sta,f])

#calculate HRV features form the BVP signal
def extract_hrv_features_bvp(sample):

    feats_hrv,filtered_sensor_data,_rr,timings,peak_indx= HRV.get_HRV_features(sample,low_pass=True,
                           ma=True,winsorize=False,dynamic_threshold_value=1,sampling=64)
    
    timings = HRV.timestamps_from_RR(_rr)

    #feats_hrv = np.concatenate([feats_hrv,[len(_rr)]])
                                                        
    return _rr,np.array(feats_hrv)

#calculate HRV features form the IBI signal
def extract_hrv_features_ibi(_rr,plt_flag=False):
#     print('extract_hrv_features_ibi',len(_rr))
    rr,outlier_indeces = HRV.hampel_filtering(_rr)
    timings = HRV.timestamps_from_RR(rr)
    timings_f,rr = HRV.medianFilter_no_idx(timings,rr)
#     print('extract_hrv_features_ibi filtered',len(rr))
    if len(rr)>=15:
        hrv_time_features = HRV.HRV_time(rr,print_flag= plt_flag)
    else:
        hrv_time_features = np.array([-1]*len(HRV.feature_names_time)) #missing values: -1
  
    #hrv_time_features = np.concatenate([hrv_time_features,[len(_rr)]])
    
                              
    return rr,np.array(hrv_time_features)


def extract_correlation_features(signal_eda,signal_st):
    corr = pearsonr(signal_eda.values,signal_st.values)[0]
    return [corr]

def extract_hr_features(signal):
    f = get_stat_features_pd_all(pd.Series(signal))
    return f

def extract_features_all(user_full_path):
    acc_freq = 32
    bvp_freq = 64
    st_freq = 4
    eda_freq = 4
    hr_freq = 1

    window_slide = 3 #seconds
    window_size = 120 #seconds

    #read BVP data from file
    bvp_df = pd.read_csv(user_full_path+'BVP.csv',header=None)
    timestamp_start = bvp_df.iloc[0].values[0]
    timestamp_end = timestamp_start+(len(bvp_df)-2)/bvp_freq
    timestamp_arr = np.arange(timestamp_start,timestamp_end,1/bvp_freq)
    bvp_df = bvp_df.iloc[2:]
    bvp_df['timestamp']=timestamp_arr
    bvp_df.columns = ['bvp','timestamp']
    bvp_df = bvp_df[['timestamp','bvp']]
    
    #read EDA data from file
    eda_df = pd.read_csv(user_full_path+'EDA.csv',header=None)
    timestamp_start = eda_df.iloc[0].values[0]
    timestamp_end = timestamp_start+(len(eda_df)-2)/eda_freq
    timestamp_arr = np.arange(timestamp_start,timestamp_end,1/eda_freq)
    eda_df = eda_df.iloc[2:]
    eda_df['timestamp']=timestamp_arr
    eda_df.columns = ['eda','timestamp']
    eda_df = eda_df[['timestamp','eda']]
    
    #read ST data from file
    st_df = pd.read_csv(user_full_path+'TEMP.csv',header=None)
    timestamp_start = st_df.iloc[0].values[0]
    timestamp_end = timestamp_start+(len(st_df)-2)/st_freq
    timestamp_arr = np.arange(timestamp_start,timestamp_end,1/st_freq)
    st_df = st_df.iloc[2:]
    st_df['timestamp']=timestamp_arr
    st_df.columns = ['st','timestamp']
    st_df = st_df[['timestamp','st']]
    
    
    #read HR data from file
    hr_df = pd.read_csv(user_full_path+'HR.csv',header=None)
    timestamp_start = hr_df.iloc[0].values[0]
    timestamp_end = timestamp_start+(len(hr_df)-2)/hr_freq
    timestamp_arr = np.arange(timestamp_start,timestamp_end,1/hr_freq)
    hr_df = hr_df.iloc[2:]
    hr_df['timestamp']=timestamp_arr
    hr_df.columns = ['hr','timestamp']
    hr_df = hr_df[['timestamp','hr']]


    #read ACC data from file
    acc_df = pd.read_csv(user_full_path+'ACC.csv',header=None)
    timestamp_start = acc_df.iloc[0,0]
    timestamp_end = timestamp_start+(len(acc_df)-2)/acc_freq
    timestamp_arr = np.arange(timestamp_start,timestamp_end,1/acc_freq)
    acc_df = acc_df.iloc[2:]
    acc_df['timestamp']=timestamp_arr
    acc_df.columns = ['x','y','z','timestamp']
    acc_df = acc_df[['timestamp','x','y','z']]
    
    #read IBI data from file
    ibi_df = pd.read_csv(user_full_path+'IBI.csv')
    time_start=ibi_df.columns[0]
    ibi_df.columns = ['relative_time','ibi']
    ibi_df['timestamp'] = float(time_start)+ ibi_df.relative_time


    print("Sessions duration in hours:", 
          round(bvp_df.shape[0]/bvp_freq/3600,2),
          round(eda_df.shape[0]/eda_freq/3600,2),
          round(st_df.shape[0]/st_freq/3600,2),
          round(acc_df.shape[0]/acc_freq/3600,2))

    features_list = []
    
    bvp_s_list = []
    st_s_list = []
    gsr_s_list = []
    labels_list = []

    #perform windowing using timestamps
    win_start = max(bvp_df.timestamp.iloc[0],eda_df.timestamp.iloc[0], st_df.timestamp.iloc[0],acc_df.timestamp.iloc[0])
    file_end = min(bvp_df.timestamp.iloc[-1],eda_df.timestamp.iloc[-1], st_df.timestamp.iloc[-1],acc_df.timestamp.iloc[-1])
    win_end = win_start+window_size #seconds
 
    while(win_end<=file_end):
        try:
            #get windows/segments
            bvp_sample = bvp_df[(bvp_df.timestamp>=win_start) & (bvp_df.timestamp<=win_end)]
            eda_sample = eda_df[(eda_df.timestamp>=win_start) & (eda_df.timestamp<=win_end)]
            st_sample = st_df[(st_df.timestamp>=win_start) & (st_df.timestamp<=win_end)]
            acc_sample = acc_df[(acc_df.timestamp>=win_start) & (acc_df.timestamp<=win_end)]
            ibi_sample = ibi_df[(ibi_df.timestamp>=win_start) & (ibi_df.timestamp<=win_end)]
            hr_sample = hr_df[(hr_df.timestamp>=win_start) & (hr_df.timestamp<=win_end)]
                        
#             print(bvp_sample.shape,eda_sample.shape,st_sample.shape, acc_sample.shape)
            #extracct features
            bvp_f = extract_bvp_features(bvp_sample.bvp)
            st_f = extract_st_features(st_sample.st)
            eda_f = extract_eda_features(eda_sample.eda)
            acc_f = extract_acc_features(acc_sample[['x','y','z']])
            rr,hrv_feats=extract_hrv_features_bvp(bvp_sample.bvp)
            rr,hrv_feats_ibi= extract_hrv_features_ibi(ibi_sample.ibi.values)
            corr_f = extract_correlation_features(eda_sample.eda,st_sample.st)
            hr_f = extract_hr_features(hr_sample.hr)

            info_ = [win_start,win_end]
#             print(len(info_),bvp_f.shape,st_f.shape,eda_f.shape,acc_f.shape,hrv_feats.shape,hrv_feats_ibi.shape,len(corr_f))

            line = np.concatenate((info_,bvp_f,st_f,eda_f,acc_f,hrv_feats,hrv_feats_ibi,corr_f,hr_f))
            features_list.append(line)
#             print(len(line))

            win_start = win_start+window_slide
            win_end = win_start+window_size
        except Exception as e:
            win_start = win_start+window_slide
            win_end = win_start+window_size
            print(e)

    return np.stack(features_list)

def plot_features_per_session(df,plots_path):
    sns.set(font_scale=1.2)
    
    plt_features = ['hr_mean','HRV_BVP_sdnn',
                        'st_mean','eda_stat_mean',
                        'eda_num_peaks','acc_magnitude']

    for s in df.uid.unique():
        plt.figure()
        s_df = df[df.uid==s][plt_features]
        s_df = s_df[s_df>0]
        s_df.plot(subplots=True,figsize=(20,9))
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        plt.savefig(plots_path+s+'.jpg')
    
        # plt.show()


def calculate_all_features(data_path):
    plots_path = data_path + 'plots/'

    df_arr = []
    tmp_date = '2022-05-20'
    min_date_time = pd.to_datetime(tmp_date) + pd.Timedelta('1days')
    max_date_time = pd.to_datetime(tmp_date) - pd.Timedelta('1days')

    column_names = ['win_start','win_end']
    column_names.extend(feature_names)


    for user in os.listdir(data_path):
        user_full_path = data_path+'/'+user+'/'
        participant_id,session_time = user.split('_')
        date = str(datetime.strptime(session_time.split('-')[0], "%d%m%y").date())
        if not os.path.isdir(user_full_path) or 'plots' in user_full_path:
            continue
        print(user)
        participant = participant_id
        print('Extracting features from:',date,participant)

        try:
            features = extract_features_all(user_full_path)
            df =pd.DataFrame(features,columns =column_names)
            #convert timestamps to dateTime and conert to Slovenian Timezone
            df['dateTime_start']= pd.to_datetime(df.win_start,unit ='s').dt.tz_localize('UTC').dt.tz_convert('Europe/Brussels').dt.strftime('%Y-%m-%d %H:%M:%S')
            df['dateTime_end']= pd.to_datetime(df.win_end,unit ='s').dt.tz_localize('UTC').dt.tz_convert('Europe/Brussels').dt.strftime('%Y-%m-%d %H:%M:%S')

            #swap columns
            new_columns = [df.columns[0]]
            new_columns.extend(df.columns[-2:])
            new_columns.extend(df.columns[1:-2])
            df = df[new_columns]

            #mark null values with -1
            #df[df.isnull()]=-1
            df = df.where(df.notnull(), -1)
            #remove duplicate features
            remove_column = ['eda_meanGsr', 'eda_stdGsr','eda_q25Gsr', 'eda_q75Gsr', 'eda_qdGsr', 'eda_derivGsr']
            df = df.drop(remove_column,axis=1)

            df['participant'] = participant
            df['dateTime_start'] = pd.to_datetime(df['dateTime_start'])
            df['dateTime_end'] = pd.to_datetime(df['dateTime_end'])
      
            #save the minimum data_time to perform resampling with equal windows
            if min_date_time>= df['dateTime_start'].min():
                min_date_time = df['dateTime_start'].min()
            if max_date_time<= df['dateTime_start'].max():
                max_date_time = df['dateTime_start'].max()

            df_arr.append(df)
        except Exception as e:
            print("Failed for: ", data_path, user_full_path)
            print("Error:", e)

    #crate a range of timestamps for resampling
    current_date = min_date_time
    rolling_delta = pd.Timedelta('10s')
    range_dates = []
    while current_date<=max_date_time:
        range_dates.append(current_date)
        current_date = current_date+rolling_delta

    feature_columns = df.columns[3:-1] #for which features to performing rolling mean
    HRV_columns = [] #These are most susceptible to noise and should be treated differently during resampling
    non_HRV_columns = []
    for f in feature_columns:
        if 'HRV' in f:
            HRV_columns.append(f)
        else:
            non_HRV_columns.append(f)
    new_columns_names = ['uid', 'dateTime_start', 'dateTime_end']
    new_columns_names.extend(non_HRV_columns)
    new_columns_names.extend(HRV_columns)

    #perform resampling
    df_resampled = []
    for df in df_arr:
        df_line_arr = []
        for i in range(len(range_dates)-1):
            start = range_dates[i]
            end = range_dates[i+1]
            select_idx = (df.dateTime_start>=start) & (df.dateTime_start<=end)
            if sum(select_idx)>0:
                values_non_HRV  =  df[select_idx][non_HRV_columns].astype(float).median()
                values_HRV =  df[select_idx][HRV_columns]
                values_HRV[values_HRV==-1] = np.nan #-1 represtnts missing values or noisy data for HRV
                values_HRV = values_HRV[values_HRV!=-1].astype(float).median() #ignores nan

                #session_id = df.session_id[0]
                dateTime_start = start
                dateTime_end = end
                participant = df.participant[0]
                line =  [participant,dateTime_start,dateTime_end] 
                line.extend(values_non_HRV)
                line.extend(values_HRV)
                df_line_arr.append(line)

        df_line_arr = pd.DataFrame(df_line_arr,columns = new_columns_names)
        df_resampled.append(df_line_arr)
    df_resampled = pd.concat(df_resampled)
    
    # # merge all data into the same
    df_resampled = df_resampled.set_index('dateTime_start')
    plot_features_per_session(df_resampled,plots_path)
    df_resampled.to_csv(data_path+'features.csv')
    save_features = ['hr_mean','HRV_BVP_sdnn','st_mean','eda_stat_mean','acc_magnitude']
    #subsampling
    df_all = []
    for user in df_resampled.uid.unique():
        user_df = df_resampled[df_resampled.uid==user]
        user_df = user_df[save_features].resample('2min').median().round(1)
        user_df['uid'] = user
        user_df = user_df.reset_index()
        df_all.append(user_df)
    pd.concat(df_all,ignore_index=True).to_csv(data_path+'features_small.csv')
                          
