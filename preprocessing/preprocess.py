import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import datetime

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys

argv = sys.argv[1:]

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

holidays_AT = holidays.country_holidays('AT')

'''
#Coloumn names english translation:
electricity_usage = 'electricity_usage'
fbh_kalte = 'fbh_kalte'
fbh_warme = 'fbh_warme'
fernwarme = 'district_heating'
luftung_kalte = 'vent_cooling'
luftung_warme = 'vent_heating'
pv = 'pv_production'
turnsaal_warme = 'gym_heating'
warm_wasser = 'water_heating'
'''

def parse_weather_timestamp(timestamp):
    return(timestamp[6:8]+ '.' + timestamp[4:6] + '.' + timestamp[2:4] + ' ' + timestamp[9:11] + ':' + timestamp[11:])

def timestamp_to_datetime(timestamp):
    date_str = timestamp.split(' ')[0]
    time_str = timestamp.split(' ')[1]
    datetime_obj = datetime.datetime(int('20'+date_str.split('.')[2]), int(date_str.split('.')[1] ), int(date_str.split('.')[0]),int(time_str[:2]), int(time_str[-2:]))
    return datetime_obj

def get_is_schoolday(date_arg):
    '''
    arg: datetime object
    returns 1: if it is a school day
    returns 0: if date is either in the weekend, a public holiday or during school break in styria
    '''
    #initilising to bad dates
    semester_break_start = datetime.date(1999,1,1)
    semester_break_end = datetime.date(1999,1,1)
    easter_start = datetime.date(1999,1,1)
    easter_end = datetime.date(1999,1,1)
    pentecost_start = datetime.date(1999,1,1)
    pentecost_end = datetime.date(1999,1,1)
    summer_start = datetime.date(1999,1,1)
    summer_end = datetime.date(1999,1,1)
    autumn_start = datetime.date(1999,1,1)
    autumn_end = datetime.date(1999,1,1)
    christmas_start = datetime.date(1999,1,1)
    christmas_end = datetime.date(1999,1,1)

    if date_arg.year == 2021:
        semester_break_start = datetime.date(2021,2,15)
        semester_break_end = datetime.date(2021,2,21)
        easter_start = datetime.date(2021,3,27)
        easter_end = datetime.date(2021,4,5)
        pentecost_start = datetime.date(2021,5,22)
        pentecost_end = datetime.date(2021,5,24)
        summer_start = datetime.date(2021,7,10)
        summer_end = datetime.date(2021,9,12)
        autumn_start = datetime.date(2021,10,27)
        autumn_end = datetime.date(2021,10,31)
        christmas_start = datetime.date(2021,12,24)
        christmas_end = datetime.date(2022,1,6)
       
    elif date_arg.year == 2022:
        semester_break_start = datetime.date(2022,2,21)
        semester_break_end = datetime.date(2022,2,21)
        easter_start = datetime.date(2022,4,9)
        easter_end = datetime.date(2022,4,18)
        pentecost_start = datetime.date(2022,6,4)
        pentecost_end = datetime.date(2022,6,6)
        summer_start = datetime.date(2022,7,9)
        summer_end = datetime.date(2022,9,11)
        autumn_start = datetime.date(2022,10,27)
        autumn_end = datetime.date(2022,10,31)
        christmas_start = datetime.date(2022,12,24)
        christmas_end = datetime.date(2023,1,7)

    
    if semester_break_start <= date_arg.date() <= semester_break_end:
        return 0
    elif easter_start <= date_arg.date() <= easter_end:
        return 0
    elif pentecost_start <= date_arg.date() <= pentecost_end:
        return 0
    elif summer_start <= date_arg.date() <=  summer_end:
        return 0
    elif autumn_start <= date_arg.date() <= autumn_end:
        return 0
    elif christmas_start <= date_arg.date() <= christmas_end:
        return 0
    elif date_arg.date() in holidays_AT:
        return 0
    elif 5 <= date_arg.weekday() <= 6:
        
        return 0

    else:
        return 1
    

def preprocess(data_path,feature):

    #Loading weather data
    graz_weather_df = pd.read_csv("../data/graz_weather.csv",delimiter=',',header=9,encoding='UTF-8')
    graz_weather_df = graz_weather_df[['timestamp', 'Graz Temperature [2 m elevation corrected]','Graz Shortwave Radiation', \
    'Graz Direct Shortwave Radiation', 'Graz Diffuse Shortwave Radiation','Graz Relative Humidity [2 m]']]

    graz_weather_df['timestamp'] = graz_weather_df['timestamp'].apply(parse_weather_timestamp).apply(timestamp_to_datetime)

    energy_df = pd.read_csv(data_path,delimiter=';',names=['timestamp',feature,'',' '], skiprows=9,encoding='unicode_escape',on_bad_lines='skip')

    #Dropping the last n=9 rows as they are not part of the dataset
    n = 9
    energy_df.drop(energy_df.tail(n).index, inplace = True)

    energy_df = energy_df[['timestamp',feature]]
    
    df = pd.DataFrame([])
    df['timestamp'] = energy_df['timestamp']
    df['timestamp'] = df['timestamp'].apply(timestamp_to_datetime)
    df[feature] = pd.to_numeric(energy_df[feature].str.replace(',','.'))

    graz_weather_df = graz_weather_df.set_index(graz_weather_df['timestamp'])
    df = df.set_index(df['timestamp'])


    # Interpolating weather data to be sampled every 15 mins

    graz_weather_df_upsampled = graz_weather_df[['Graz Temperature [2 m elevation corrected]',
       'Graz Shortwave Radiation', 'Graz Direct Shortwave Radiation',
       'Graz Diffuse Shortwave Radiation', 'Graz Relative Humidity [2 m]']].resample('15T').interpolate()

    #graz_weather_df_upsampled.reset_index(drop = True, inplace = True)
    df.reset_index(drop = True, inplace = True)
    
    #merging the datasets
    df = pd.merge(df,graz_weather_df_upsampled,on='timestamp',how='inner')
    df = df.dropna()

    df.index = df['timestamp']

    df['is_schoolday'] = df['timestamp'].apply(get_is_schoolday)

    #Splting timestamp into day, month, year, hour
    df['day'] = [x.day for x in df['timestamp']]
    df['month'] = [x.month for x in df['timestamp']]
    df['year'] = [x.year for x in df['timestamp']]
    df['hour'] = [x.hour for x in df['timestamp']]

    df.to_csv('../data/preprocessed/'+feature+'_weather'+'.csv',index=False)


preprocess(argv[0],argv[1])