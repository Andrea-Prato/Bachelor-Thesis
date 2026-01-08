import pandas as pd
import GSR_analysis as GSR
import HRV_analysis as HRV
import numpy as np
import os
from scipy.stats import pearsonr
from scipy.stats import zscore
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
    
def plot_ISC(df,name,plots_path):
    hue_order = df.participant.unique()
    hue_order.sort()
    plt.figure(figsize=(20,2.5))
    sns.lineplot(data=df,x='dateTime_start',y='corr_r',hue='participant',style='participant',
                     alpha=.4,palette='mako_r',hue_order=hue_order)
    sns.lineplot(data=df,x='dateTime_start',y='corr_r',label='Mean')
    y_name = 'ISC-'+name
    plt.ylabel(y_name)
    plt.xlabel('Date-time')
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    plt.savefig(plots_path+y_name+'.jpg')
#     plt.show()
    return
    
def interpolate_HR(rr_intervals,peak_indices):
    bvp_freq = 64
    new_sanmpling_rate = 4
    start_time = 8 #leave-out first n seconds. First 2 seconds are already missing because of fitlering
    duration = 60 #seconds
    end_time = 60 + start_time #  60 seconds of data
    
    x_equi = np.linspace(start_time,end_time,num=new_sanmpling_rate*duration)
    x = peak_indices/bvp_freq #convert indiced to seconds
    y = 60/rr_intervals #from RR intervals to heart rate
    f_interp = interp1d(x,y,kind='quadratic')
    HR_equi = f_interp(x_equi) #equdistant heart rate sampled at rate new_sanmpling_rate
    
    return HR_equi

#Inter-subject correlation (ISC-HR) based on instantaneous HR
#from Pérez, Pauline, et al. "Conscious processing of narrative stimuli synchronizes heart rate between individuals." Cell Reports 36.11 (2021): 109692.

def calculate_ISC(synced_df):
    data_columns = synced_df.columns[4:] # first 4 columns are meta data
    corr_data = []
    print('Timestamps total: ', synced_df.shape[0])
    print('Segments total: ', len(synced_df.start_timestamp.unique()))
    for t_start in synced_df.start_timestamp.unique():
        segment_data = synced_df[synced_df.start_timestamp==t_start].copy() #specific segment for all subjects
        if len(segment_data.participant.unique())==1:
            continue
        segment_data.loc[:,data_columns] = zscore(segment_data.loc[:,data_columns],axis=1) #row-wise z score
        segment_data.set_index('participant',inplace=True)

        corr_matrix = segment_data[data_columns].T.corr() #pearson's correlation each row with each row

        #mean correlation excluding self-correlation
        corr_matrix = np.arctanh(corr_matrix) #fisher transformation
        np.fill_diagonal(corr_matrix.values, -2)
        corr_matrix =  corr_matrix[corr_matrix!=-2].mean() #mean correlation per participant
        corr_matrix = np.tanh(corr_matrix) #inverse fisher transformation
        corr_matrix = pd.DataFrame(corr_matrix).reset_index()
        
        corr_matrix.columns = ['participant','corr_r'] #add meta-data
        corr_matrix['start_timestamp'] = t_start
        corr_matrix['end_timestamp'] = segment_data.end_timestamp[0]
        corr_data.append(corr_matrix)
    corr_data = pd.concat(corr_data)  
    
    #median rasampling using 5 minute intervals for each participant
    resampled_corr = []
    for p in corr_data.participant.unique():
        p_df= corr_data[corr_data.participant==p]
        p_df.start_timestamp=  pd.to_datetime(p_df.start_timestamp,unit='s')
        
        p_df = p_df.set_index('start_timestamp') 
        p_df = p_df.resample('5min').median()
        p_df['participant'] = p
        
        
        p_df.end_timestamp=  p_df.index + pd.Timedelta(minutes=5)
        p_df = p_df.reset_index()
        #conveert to swiss time
        p_df.start_timestamp = p_df.start_timestamp.dt.tz_localize('UTC').dt.tz_convert('Europe/Brussels')
        p_df.end_timestamp = p_df.end_timestamp.dt.tz_localize('UTC').dt.tz_convert('Europe/Brussels')
#         resampled_corr.append(p_df.reset_index())
        resampled_corr.append(p_df)
    resampled_corr = pd.concat(resampled_corr).sort_values('start_timestamp')
    resampled_corr = resampled_corr[['start_timestamp','end_timestamp', 'corr_r', 'participant']] #reorder columns
    resampled_corr.columns=['dateTime_start','dateTime_end', 'corr_r', 'participant'] #rename columns
    resampled_corr.reset_index(inplace=True, drop=True)

    return resampled_corr

def calculate_all_ISCs(folder_path):
    bvp_freq = 64
    eda_freq = 4

    bvp_dict_all = {} #readd the BVP data from all participants
    eda_dict_all = {} #readd the EDA data from all participants

    print(folder_path)
    plots_path = os.path.join(folder_path, 'plots')
    
    video_timestamps = pd.read_csv(os.path.join(folder_path, '../../timestamps.csv'))
    
    video_types = list(video_timestamps.video.unique())+['break_%d' % n for n in range(0,6)]
    
    min_timestamps = dict.fromkeys(video_types, np.inf)
    max_timestamps = dict.fromkeys(video_types, -np.inf)

    for user_full_path in [os.path.join(folder_path, folder) for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]:
        if user_full_path.endswith('plots'): continue

        user = user_full_path.split('/')[-1]
        participant_id, session_time = user.split('_')

        user_full_path += '/'
        date = str(datetime.strptime(session_time.split('-')[0], "%d%m%y").date())

        #read BVP and GSR data from the session, now user
        bvp_df = pd.read_csv(user_full_path+'BVP.csv',header=None)
        eda_df = pd.read_csv(user_full_path+'EDA.csv',header=None)
        
        print("I opened the %sBVP.csv file" % user_full_path)
       
        #format BVP data
        timestamp_start = bvp_df.iloc[0].values[0]
        timestamp_end = timestamp_start+(len(bvp_df)-2)/bvp_freq
        timestamp_arr = np.arange(timestamp_start,timestamp_end,1/bvp_freq) #timestamps in seconds 
        bvp_df = bvp_df.iloc[2:]
        bvp_df['timestamp_absolute']=timestamp_arr
        bvp_df.columns = ['bvp','timestamp_absolute']
        bvp_df = bvp_df[['timestamp_absolute','bvp']]
        bvp_df['participant'] = participant_id
        bvp_df['timestamp'] = pd.Series(dtype='float64')
        bvp_df['video'] = pd.Series(dtype=str)
        
        #format GSR data
        timestamp_start = eda_df.iloc[0].values[0]
        timestamp_end = timestamp_start+(len(eda_df)-2)/eda_freq
        timestamp_arr = np.arange(timestamp_start,timestamp_end,1/eda_freq) #timestamps in seconds 
        eda_df = eda_df.iloc[2:]
        eda_df['timestamp_absolute']=timestamp_arr
        eda_df.columns = ['eda','timestamp_absolute']
        eda_df = eda_df[['timestamp_absolute','eda']]
        eda_df['participant'] = participant_id
        eda_df['timestamp'] = pd.Series(dtype='float64')
        eda_df['video'] = pd.Series(dtype=str)
        
        break_no = 0
        for index, row in video_timestamps.loc[video_timestamps.participant == participant_id].iterrows():
          # Fill NaN before current video row with break_N
          # Actually incorrect, as it includes "part 1" of the experiment, but we are not considering that here
            eda_df.loc[(eda_df.timestamp_absolute < datetime.strptime(row.timestamp_start, '%Y-%m-%d %H:%M:%S').timestamp()), 'video'] = eda_df.loc[(eda_df.timestamp_absolute < datetime.strptime(row.timestamp_start, '%Y-%m-%d %H:%M:%S').timestamp()), 'video'].fillna('break_%d' % break_no)
            bvp_df.loc[(bvp_df.timestamp_absolute < datetime.strptime(row.timestamp_start, '%Y-%m-%d %H:%M:%S').timestamp()), 'video'] = bvp_df.loc[(bvp_df.timestamp_absolute < datetime.strptime(row.timestamp_start, '%Y-%m-%d %H:%M:%S').timestamp()), 'video'].fillna('break_%d' % break_no)
            break_no += 1
          
            # Fill NaN during current video row with the correct video type
            eda_df.loc[(eda_df.timestamp_absolute >= datetime.strptime(row.timestamp_start, '%Y-%m-%d %H:%M:%S').timestamp()) & (eda_df.timestamp_absolute <= datetime.strptime(row.timestamp_end, '%Y-%m-%d %H:%M:%S').timestamp()), 'video'] = row.video
            bvp_df.loc[(bvp_df.timestamp_absolute >= datetime.strptime(row.timestamp_start, '%Y-%m-%d %H:%M:%S').timestamp()) & (bvp_df.timestamp_absolute <= datetime.strptime(row.timestamp_end, '%Y-%m-%d %H:%M:%S').timestamp()), 'video'] = row.video

        # Fill NaN after last video row
        eda_df.loc[(eda_df.timestamp_absolute > datetime.strptime(row.timestamp_end, '%Y-%m-%d %H:%M:%S').timestamp()), 'video'] = eda_df.loc[(eda_df.timestamp_absolute > datetime.strptime(row.timestamp_end, '%Y-%m-%d %H:%M:%S').timestamp()), 'video'].fillna('break_%d' % break_no)
        bvp_df.loc[(bvp_df.timestamp_absolute > datetime.strptime(row.timestamp_end, '%Y-%m-%d %H:%M:%S').timestamp()), 'video'] = bvp_df.loc[(bvp_df.timestamp_absolute > datetime.strptime(row.timestamp_end, '%Y-%m-%d %H:%M:%S').timestamp()), 'video'].fillna('break_%d' % break_no)
        
        for video in eda_df.video.unique():
            eda_df.loc[eda_df.video == video, 'timestamp'] = eda_df.loc[eda_df.video == video, 'timestamp_absolute'] - eda_df.loc[eda_df.video == video].iloc[0].timestamp_absolute
            bvp_df.loc[bvp_df.video == video, 'timestamp'] = bvp_df.loc[bvp_df.video == video, 'timestamp_absolute'] - bvp_df.loc[bvp_df.video == video].iloc[0].timestamp_absolute
        
            if min_timestamps[video] > bvp_df.loc[bvp_df.video == video].timestamp.iloc[0]:
                min_timestamps[video] = bvp_df.loc[bvp_df.video == video].timestamp.iloc[0]
            if max_timestamps[video] < bvp_df.loc[bvp_df.video == video].timestamp.iloc[-1]:
                max_timestamps[video] = bvp_df.loc[bvp_df.video == video].timestamp.iloc[-1]
            
        #drop first 5 seconds (bad signal)
        bvp_df = bvp_df.iloc[5*bvp_freq:,:]
        #key = bvp_df.participant.iloc[0]+'_'+bvp_df.session.iloc[0]
        key = bvp_df.participant.iloc[0]
        bvp_dict_all[key] = bvp_df

        #drop first 5 seconds (bad signal)
        eda_df = eda_df.iloc[5*eda_freq:,:]
        #key = eda_df.participant.iloc[0]+'_'+eda_df.session.iloc[0]
        key = eda_df.participant.iloc[0]
        eda_dict_all[key] = eda_df


    window_slide = 5 #seconds
    window_size = 80 #seconds 
    
    for video in eda_df.video.unique():
      #perform windowing using timestamps
        print("Syncing video %s" %video)
        win_start = min_timestamps[video]
        win_end = win_start+window_size
        synced_HR = []
        synced_EDA = []
        while(win_end<=max_timestamps[video]):
            line_arr = []
            line_arr_eda = []
            for key in bvp_dict_all:
                #print(video, key)
                bvp_df_tmp = bvp_dict_all[key]
                bvp_df = bvp_df_tmp[bvp_df_tmp.video == video]
                bvp_sample = bvp_df[(bvp_df.timestamp>=win_start) & (bvp_df.timestamp<=win_end)].bvp.values

                eda_df_tmp = eda_dict_all[key]
                eda_df = eda_df_tmp[eda_df_tmp.video == video] 
                eda_sample = eda_df[(eda_df.timestamp>=win_start) & (eda_df.timestamp<=win_end)].eda.values

                if len(bvp_sample) <window_size*bvp_freq:
                    #print("No data for the specific timestamps!")
                    continue # no data for the specific timestamps

                #PROCESS GSR DATA
                eda_sample = pd.Series(eda_sample).rolling(3).mean().dropna().values #moving average with 3 samples (.75 seconds)
                eda_sample = eda_sample[10*eda_freq:10*eda_freq+60*eda_freq] #take middle 60 seconds of data out of the overall 80 seconds
                subject = key.split('_')[0]
                line = [date,win_start,win_end,subject]
                line.extend(eda_sample)
                line_arr_eda.append(line)

                #PROCESS HR DATA
                try: #fails if the PPG data is noisy
                  #get RR intervals
                    feats_hrv,filtered_sensor_data,_rr,timings,peak_indx= HRV.get_HRV_features(bvp_sample,low_pass=True, ma=True,winsorize=False,dynamic_threshold_value=1,sampling=64,plt_flag=False)

                    if len(feats_hrv)==0 or feats_hrv[0]==-1: #bad sensor data
                        continue

                    HR_equi = interpolate_HR(_rr,peak_indx)
                    subject = key.split('_')[0]
                    line = [date,win_start,win_end,subject]
                    line.extend(HR_equi)
                    line_arr.append(line)
                except Exception as e:
                    #print(e)
                    continue

            synced_HR.extend(line_arr)
            synced_EDA.extend(line_arr_eda)

            win_start = win_start+window_slide
            win_end = win_start+window_size
        isc_hr = []
        if len(synced_HR)>0:
            columns=['date','start_timestamp','end_timestamp','participant']
            columns.extend(np.arange(240))
            synced_HR = pd.DataFrame(synced_HR,columns = columns) #dataframe with synced HRs
            synced_EDA = pd.DataFrame(synced_EDA,columns = columns) #dataframe with synced GSRs

            synced_HR.start_timestamp = synced_HR.start_timestamp+10 #we droped first and last 10 seconds in preprocessing so we use only 1 minute data
            synced_HR.end_timestamp = synced_HR.end_timestamp-10

            synced_EDA.start_timestamp = synced_EDA.start_timestamp+10 #we droped first and last 10 seconds in preprocessing so we use only 1 minute data
            synced_EDA.end_timestamp = synced_EDA.end_timestamp-10

            print('Calculating ISC for HR...')
            isc_hr = calculate_ISC(synced_HR)
            print('Calculating ISC for EDA...')
            isc_eda = calculate_ISC(synced_EDA)


            plot_ISC(isc_eda,"EDA_%s"%video,plots_path)
            plot_ISC(isc_hr,"HR_%s"%video,plots_path)

            isc_hr.dateTime_start = isc_hr.dateTime_start.dt.strftime('%Y-%m-%d %H:%M:%S')
            isc_eda.dateTime_start = isc_eda.dateTime_start.dt.strftime('%Y-%m-%d %H:%M:%S')
            isc_hr.dateTime_end = isc_hr.dateTime_end.dt.strftime('%Y-%m-%d %H:%M:%S')
            isc_eda.dateTime_end = isc_eda.dateTime_end.dt.strftime('%Y-%m-%d %H:%M:%S')

            isc_hr.to_csv(folder_path+'ISC_HR_%s.csv'%video)
            isc_eda.to_csv(folder_path+'ISC_EDA_%s.csv'%video)
    return

