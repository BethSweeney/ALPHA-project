import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import allantools as allantools
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import os
import re
from math import ceil, floor, log10, sqrt, trunc, modf
import sympy as sy
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from typing import NamedTuple
from datetime import datetime, timedelta
import copy
from scipy import optimize, integrate
from ftplib import FTP, error_perm
from io import BytesIO
from allantools import oadev
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from matplotlib.figure import Figure
import time
import warnings
warnings.filterwarnings("ignore")
import threading
from threading import Timer


save_dir=r'C:\Users\bethx\OneDrive\Desktop'

## Define Functions to read in data ###########################################

def mjd_now():
    """
    Convert a date to Modified Julian Day.

    Algorithm sourced from 'Practical Astronomy with your Calculator or Spreadsheet',
        4th ed., Duffet-Smith and Zwart, 2011.

    Parameters
    ----------
    year : int
        Year as integer. Years preceding 1 A.D. should be 0 or negative.
        The year before 1 A.D. is 0, 10 B.C. is year -9.

    month : int
        Month as integer, Jan = 1, Feb. = 2, etc.

    day : float
        Day, may contain fractional part.

    Returns
    -------
    mjd : float
        Modified Julian Day

    """
   
    year, month, day, h, m, s = datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour, datetime.now().minute, datetime.now().second,
   
    if month == 1 or month == 2:  # months rescaled so february is last month of the year because 28 days
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month

    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if ((year < 1582) or
            (year == 1582 and month < 10) or
            (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
        print('Time must be later than October 1582')
        raise SystemExit

    # after start of Gregorian calendar
    a = trunc(yearp / 100.)  # discount leap years falling on a multiple of 100
    b = 2 - a + trunc(a / 4.)  # recount leap years falling on a multiple of 400

    c = trunc(365.25 * yearp)

    d = trunc(30.6001 * (monthp + 1))

    jd = b + c + d + day + 1720994.5  # Integer is julian day for 2 B.C. Oct 30
    jd += h/24 + m/1440 + s/86400

    return jd - 2400000.5

def date_to_mjd(year, month, day):
    """
    Convert a date to Modified Julian Day.

    Algorithm sourced from 'Practical Astronomy with your Calculator or Spreadsheet',
        4th ed., Duffet-Smith and Zwart, 2011.

    Parameters
    ----------
    year : int
        Year as integer. Years preceding 1 A.D. should be 0 or negative.
        The year before 1 A.D. is 0, 10 B.C. is year -9.

    month : int
        Month as integer, Jan = 1, Feb. = 2, etc.

    day : float
        Day, may contain fractional part.

    Returns
    -------
    mjd : float
        Modified Julian Day

    """
    if month == 1 or month == 2:  # months rescaled so february is last month of the year because 28 days
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month

    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if ((year < 1582) or
            (year == 1582 and month < 10) or
            (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
        print('Time must be later than October 1582')
        raise SystemExit

    # after start of Gregorian calendar
    a = trunc(yearp / 100.)  # discount leap years falling on a multiple of 100
    b = 2 - a + trunc(a / 4.)  # recount leap years falling on a multiple of 400

    c = trunc(365.25 * yearp)

    d = trunc(30.6001 * (monthp + 1))

    jd = b + c + d + day + 1720994.5  # Integer is julian day for 2 B.C. Oct 30

    return jd - 2400000.5


def mjd_to_date(mjd):
    """
    Convert Modified Julian Day to date.

    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
        4th ed., Duffet-Smith and Zwart, 2011.

    Parameters
    ----------
    mjd : float
        Julian Day

    Returns
    -------
    year : int
        Year as integer. Years preceding 1 A.D. should be 0 or negative.
        The year before 1 A.D. is 0, 10 B.C. is year -9.

    month : int
        Month as integer, Jan = 1, Feb. = 2, etc.

    day : float
        Day, may contain fractional part.
    """
    jd = mjd + 2400001

    F, I = modf(jd)
    I = int(I)

    A = trunc((I - 1867216.25) / 36524.25)

    if I > 2299160:
        B = I + 1 + A - trunc(A / 4.)
    else:
        B = I

    C = B + 1524

    D = trunc((C - 122.1) / 365.25)

    E = trunc(365.25 * D)

    G = trunc((C - E) / 30.6001)

    day = C - E + F - trunc(30.6001 * G)

    if G < 13.5:
        month = G - 1
    else:
        month = G - 13

    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715

    month = '{:02}'.format(month)

    return year, month, day

def fountain_stats(HD_data, LD_data, HD_adev, LD_adev):
    """
     Returns
    -------
    f0 : float - zero-density fountain frequency
    uA_diff: uncertainty on the freq diff betwwen high and low density
    uA_f0: statistical uncertainty on f0
    uB_f0: systematic uncertainty on collisional shift
    uA_HD/uA_LD: statistical uncertainty of the high-density mean fractional frequency difference,
           excluding dead time effects. It is given by the successive-point Allan deviation of the high-density
           measurements divided by the square root of the number of high density points in the time window
    fHD/fLD: average fractional frequency difference for data set
    """

    if len(HD_data):
        fHD = HD_data['Frequency Error'].mean()
        nHD = HD_data['Shot A Sum'].mean()
        uA_HD = HD_adev / sqrt(len(HD_data.dropna()))
    else:
        fHD, nHD, uA_HD = np.nan, np.nan, np.nan

    if len(LD_data):
        fLD = LD_data['Frequency Error'].mean()
        nLD = LD_data['Shot A Sum'].mean()
        uA_LD = LD_adev / sqrt(len(LD_data.dropna()))
    else:
        fLD, nLD, uA_LD = np.nan, np.nan, np.nan
       

    uA_diff = sqrt(uA_HD ** 2 + uA_LD ** 2)

    f0 = fLD - (nLD * ((fHD - fLD) / (nHD - nLD))) # exptrapolate to zero density
   
   
    uA_f0 = sqrt((uA_HD * (nLD / (nHD - nLD))) ** 2 + (uA_LD * (1 + (nLD / (nHD - nLD)))) ** 2)

    uB_f0 = 0.1 * ((nHD * nLD) / ((nHD - nLD) ** 2)) * sqrt((fHD - fLD) ** 2 + uA_diff ** 2)

    return [f0 / 9192631770, uA_f0, uB_f0 / 9192631770, uA_diff, uA_HD,
                              uA_LD, fHD / 9192631770, fLD / 9192631770]


def overlapping_allan_dev(data, rate=0.3):
    max_tau = np.log10(len(data) / 0.6)
    taus = np.logspace(0, max_tau, 40)
    y = np.array(data.dropna())  # Prepare the frequency data
    allan_tuple = allantools.oadev(y, rate=rate, data_type="freq", taus=taus)  # Get overlapping ADEV
    return allan_tuple

def add_error_quadrature(err1, err2):  # Takes floats and returns the quadrature as a scientific string
    tot = np.sqrt(float(err1) ** 2 + float(err2) ** 2)
    return np.format_float_scientific(tot, precision=3, trim='0')

def fountain_hourly_write(HD, LD, start_date_mjd, end_date_mjd):
    def find_deadtimes(h_data, h):
        # Make list of deadtime ranges in seconds from start of bin paired in tuples
        seconds = h_data['Seconds'].astype(int)
        start_index = np.flatnonzero(np.diff(seconds) > 5)
        end_index = start_index + 1

        dead_start = seconds.iloc[start_index]
        dead_end = seconds.iloc[end_index]
        dead_times = list(zip(dead_start, dead_end))

        dead_times = [(h*3600, seconds.iloc[0])] + dead_times
        dead_times = dead_times + [(seconds.iloc[-1], (h+1)*3600)]
        return dead_times

    data = []
    start = modf(float(start_date_mjd))
    end = modf(float(end_date_mjd))
    start_hour = ceil(start[0] / (1/24))
    end_hour = floor(end[0] / (1/24))

    start_calendar = mjd_to_date(start[1])  # type: tuple(int, int, float)
    end_calendar = mjd_to_date(end[1])  # type: tuple(int, int, float)

    start_string = f'{start_calendar[0]}-{start_calendar[1]}-{trunc(start_calendar[2])}T{start_hour:02}:00:00Z'
    end_string = f'{end_calendar[0]}-{end_calendar[1]}-{trunc(end_calendar[2])}T{end_hour:02}:00:00Z'

    DATE_TIME_STRING_FORMAT = '%Y-%m-%dT%H:%M:%SZ'

    from_date_time = datetime.strptime(start_string, DATE_TIME_STRING_FORMAT)
    to_date_time = datetime.strptime(end_string, DATE_TIME_STRING_FORMAT)

    date_times = []
    while from_date_time < to_date_time:
        date_times.append(from_date_time.strftime(DATE_TIME_STRING_FORMAT))
        from_date_time += timedelta(hours=1)


    for date_string in date_times:
        day = int(date_to_mjd(int(date_string[0:4]), int(date_string[5:7]), int(date_string[8:10])))
        hour = int(date_string[11:13])
        fountain_data, successive_adev, deadtimes = [], [], []
        for full_data in HD, LD:
            hourly_data = full_data.loc[(full_data['MJD Days'] == day)].copy(deep=False)
            hourly_data = hourly_data[hourly_data["Seconds"].between((hour * 3600) + 1, (hour + 1) * 3600)]
            if len(hourly_data) > 120:
                (t2, ad, ade, adn) = overlapping_allan_dev(hourly_data['Frequency Error'])
                successive_adev.append(ad[0] / 9192631770)  # Cs hyperfine transition
                deadtimes.append(find_deadtimes(hourly_data, hour))
            else:
                deadtimes.append([(0 + (hour * 3600), (hour + 1) * 3600)])
                successive_adev.append(np.nan)
            fountain_data.append(hourly_data)
        hourly_stats = fountain_stats(fountain_data[0], fountain_data[1],
                                      successive_adev[0], successive_adev[1])
       
        uB_tot = add_error_quadrature(2E-16, hourly_stats[2])

        data.append({'Bin Start UTC': date_string, 'Bin Start MJD Days': day, 'Bin Start Seconds':
                        hour * 3600, 'Bin End MJD Days': day, 'Bin End Seconds': (hour * 3600) + 3599,
                     'f0': hourly_stats[0], 'uA (f0)': hourly_stats[1], 'uB (f0)': float(uB_tot),
                     'Freq Diff (LD-HD)': hourly_stats[7] - hourly_stats[6],
                     'uA (LD-HD)': hourly_stats[3], 'HD Freq': hourly_stats[6],
                     'uA (HD)': hourly_stats[4], 'LD Freq': hourly_stats[7], 'uA (LD)': hourly_stats[5],
                     'Mean HD Signal (Vs)': fountain_data[0]['Shot A Sum'].mean(),
                     'uA (HD Sig)': fountain_data[0]['Shot A Sum'].sem(),
                     'Mean LD Signal (Vs)': fountain_data[1]['Shot A Sum'].mean(),
                     'uA (LD Sig)': fountain_data[1]['Shot A Sum'].sem(),
                     'No. of HD Points': len(fountain_data[0].dropna()),
                     'No. of LD Points': len(fountain_data[1].dropna()),
                     'HD Deadtimes': deadtimes[0], 'LD Deadtimes': deadtimes[1]})
    hourly_bins = pd.DataFrame.from_dict(data)
   
    return(hourly_bins)

def get_recent_frequencies(systematic_shift=4.485e-14, past_days = 15, HD_low = 30, HD_high = 90, LD_low = 4, LD_high = 15, stD = 5):
   
    #find all LD and HD file in directory
    datafiles = os.listdir(save_dir +'\Alpha_CSF_up_to_date')

    LD_files = []
    HD_files = []
    for i in datafiles:
        if(i.split()[0]=='ALPHA-CsF1'):
            if(int(float(i.split()[3])) > mjd_now() - past_days -1 and bool(i.split()[4])):
                if i.split()[6] == 'LD.txt':
                    LD_files.append(i)
                elif i.split()[6] == 'HD.txt':
                    HD_files.append(i)

    #read in files
    HD_dfs = []  
    LD_dfs = []
    

    for file in HD_files:

        HD_dfs.append(pd.read_csv(save_dir +r'\Alpha_CSF_up_to_date\\' + file , comment='%', delim_whitespace=True, header=None, usecols=[*range(0, 5)],
                                   names=['MJD Days', 'Seconds', 'Frequency Error', 'Shot A Sum', 'Shot A Ratio'],
                                   dtype={'MJD Days': 'float', 'Seconds': 'float', 'Frequency Error': 'float',
                                             'Shot A Sum': 'float', 'Shot A Ratio': 'float'}))

    for file in LD_files:
        LD_dfs.append(pd.read_csv(save_dir +r'\Alpha_CSF_up_to_date\\' + file , comment='%', delim_whitespace=True, header=None, usecols=[*range(0, 5)],
                                   names=['MJD Days', 'Seconds', 'Frequency Error', 'Shot A Sum', 'Shot A Ratio'],
                                   dtype={'MJD Days': 'float', 'Seconds': 'float', 'Frequency Error': 'float',
                                             'Shot A Sum': 'float', 'Shot A Ratio': 'float'}))
    #concentate and reset index
    HD = pd.concat(HD_dfs).reset_index()
    LD = pd.concat(LD_dfs).reset_index()

    HD['ts'] = HD['MJD Days'].astype(float) + HD['Seconds'].astype(float)/86400
    LD['ts'] = LD['MJD Days'].astype(float) + LD['Seconds'].astype(float)/86400

    HD = HD[HD['ts']>mjd_now()-past_days]
    LD = LD[LD['ts']>mjd_now()-past_days]
    

    #drops NaN's and set datatypes
    HD = HD.dropna()
    HD.astype({'MJD Days': 'int32', 'Seconds': 'int32'}).dtypes
    LD = LD.dropna()
    LD.astype({'MJD Days': 'int32', 'Seconds': 'int32'}).dtypes
   
    #remove exceedingly low and high sum values
    HD = HD[1000*HD['Shot A Sum']>HD_low]
    HD = HD[1000*HD['Shot A Sum']<HD_high]
    LD = LD[1000*LD['Shot A Sum']>LD_low]
    LD = LD[1000*LD['Shot A Sum']<LD_high]
   
    #set LD >5 SD bounds
    LD['upper bound'] = LD['Frequency Error'].rolling(150, center=True).mean() + stD*LD['Frequency Error'].rolling(150, center=True).std()
    LD['lower bound'] = LD['Frequency Error'].rolling(150, center=True).mean() - stD*LD['Frequency Error'].rolling(150, center=True).std()
    LD.loc[0:74, 'upper bound'] = LD[75:76]['upper bound'].values[0]
    LD.loc[LD.index[-1]-73:,'upper bound'] = LD[-75:-74]['upper bound'].values[0]
    LD.loc[0:74, 'lower bound'] = LD[75:76]['lower bound'].values[0]
    LD.loc[LD.index[-1]-73:,'lower bound'] = LD[-75:-74]['lower bound'].values[0]
   
    #set HD >5 SD bounds
    HD['upper bound'] = HD['Frequency Error'].rolling(50, center=True).mean() + stD*HD['Frequency Error'].rolling(50, center=True).std()
    HD['lower bound'] = HD['Frequency Error'].rolling(50, center=True).mean() - stD*HD['Frequency Error'].rolling(50, center=True).std()
    HD.loc[0:25, 'upper bound'] = HD[25:26]['upper bound'].values[0]
    HD.loc[HD.index[-1]-23:, 'upper bound'] = HD[-25:-24]['upper bound'].values[0]
    HD.loc[0:25, 'lower bound'] = HD[25:26]['lower bound'].values[0]
    HD.loc[HD.index[-1]-23:,'lower bound'] = HD[-25:-24]['lower bound'].values[0]

   
    #remove values outside those bounds
    LD_clean = LD[LD['Frequency Error']< LD['upper bound']]
    LD_clean = LD_clean[LD_clean['Frequency Error']> LD_clean['lower bound']]
    HD_clean = HD[HD['Frequency Error']< HD['upper bound']]
    HD_clean = HD_clean[HD_clean['Frequency Error']> HD_clean['lower bound']]

    #use LD and HD to get df of final freqeuncies out
    hour_bin_df = fountain_hourly_write(HD_clean, LD_clean, int(mjd_now())-past_days, int(mjd_now())+1)
   
    #add a MJD timestamp
    hour_bin_df['timestamp'] = hour_bin_df['Bin Start MJD Days'].astype(float) + hour_bin_df['Bin Start Seconds'].astype(float)/86400
    #drop missing hours
    hour_bin_df=hour_bin_df.dropna()
    #apply systematic shift
    hour_bin_df['f0'] = hour_bin_df['f0'] - systematic_shift
    return hour_bin_df

#### #############################################################



window = tk.Tk()


window.title("Suggest CSF steer :)")
window.configure(background='azure')
window.geometry('1200x700')

left_frame  = Frame(window,  width=200,  height=100, bg='azure')
left_frame.grid(row=0,  column=0, padx=30)

right_frame =  Frame(window, width=650,  height=500,bg='azure')
right_frame.grid(row=0, column=1, pady=20,sticky='w'+'e'+'n'+'s')

steering_mode_frame = Frame(window,width=600,height=200)
steering_mode_frame.grid(row=1,column=0, columnspan=2,pady=10)



Generated_data2 = pd.read_csv(r'C:\Users\bethx\OneDrive\Desktop\recent_data_w_rescaled_LD.csv')
Generated_data3 = pd.read_csv(r'C:\Users\bethx\OneDrive\Desktop\recent_data_w_rescaled_old_LD.csv')


def find_cross_section(constant,varied,constant_value,data):
    
    cross_section_x = []
    cross_section_y = []
    cross_section_z = []
    
    for i in np.arange(0,1020,1): 
                       
            x = round(data[constant][i],2)
            #print(i)
            #print(x)
                       
            if x == constant_value:
                y = data[varied][i]
                #print(y)
                z = data['Fit_max'][i]
                
                cross_section_x.append(x)
                cross_section_y.append(y)
                cross_section_z.append(z)
                
            elif x != constant_value:
                i+=1
                
    return cross_section_x,cross_section_y,cross_section_z



def fit_line(x1,y1,x2,y2,x):
    
    slope = (y2-y1)/(x2-x1)
    
    line_at_x = y1 + slope*(x-x1)
    
    return line_at_x



def interpolate_LD():

    suggest_steer_box.delete(0, END)
    entry_fit_to_Y_days_box.delete(0,END)

    try:

        RW = entry_RW_int.get()
        LD = entry_LD_int.get()
        
        RW_array,LD_array, Fit_Max_array = find_cross_section('RW','Rescaled_LD',RW ,Generated_data2)
        
        data = {'Linear Drift': LD_array,
            'Fit max': Fit_Max_array}
      
        df = pd.DataFrame(data, columns=['Linear Drift', 'Fit max'])
        
        left_of_point = df[(df['Linear Drift'])<LD]
        right_of_point = df[(df['Linear Drift'])>LD]
        
        closest_left = min(left_of_point['Linear Drift'], key =lambda x: abs(LD-x))
        closest_right = min(right_of_point['Linear Drift'], key =lambda x: abs(LD-x))
            
        index_left = (left_of_point[left_of_point['Linear Drift']==closest_left].index.values)
        index_right = (right_of_point[right_of_point['Linear Drift']==closest_right].index.values)
            
        steer_left = left_of_point['Fit max'][int(index_left)]
        steer_right = right_of_point['Fit max'][int(index_right)]
            
        interpolated_fit_max = fit_line(closest_left,steer_left,closest_right,steer_right,LD)
        
        suggest_steer_box.insert(0,round(interpolated_fit_max,4))
        entry_fit_to_Y_days_box.insert(0,round(float(suggest_steer_box.get()),4))

    
    except:

        RW = entry_RW_int.get()
        LD = entry_LD_int.get()
        
        RW_array,LD_array, Fit_Max_array = find_cross_section('RW','Rescaled_LD',RW ,Generated_data3)
        
        data = {'Linear Drift': LD_array,
            'Fit max': Fit_Max_array}
      
        df = pd.DataFrame(data, columns=['Linear Drift', 'Fit max'])
        
        left_of_point = df[(df['Linear Drift'])<LD]
        right_of_point = df[(df['Linear Drift'])>LD]
        
        closest_left = min(left_of_point['Linear Drift'], key =lambda x: abs(LD-x))
        closest_right = min(right_of_point['Linear Drift'], key =lambda x: abs(LD-x))
            
        index_left = (left_of_point[left_of_point['Linear Drift']==closest_left].index.values)
        index_right = (right_of_point[right_of_point['Linear Drift']==closest_right].index.values)
            
        steer_left = left_of_point['Fit max'][int(index_left)]
        steer_right = right_of_point['Fit max'][int(index_right)]
            
        interpolated_fit_max = fit_line(closest_left,steer_left,closest_right,steer_right,LD)
        
        suggest_steer_box.insert(0,round(interpolated_fit_max,5))
        entry_fit_to_Y_days_box.insert(0,round(float(suggest_steer_box.get()),4))


read_steer_history4 = np.loadtxt(r"C:\Users\bethx\OneDrive\Desktop\CSF_steer_history.txt")


def plot():

    fig.clear()
    output_box.delete(0, END)
    last_steer_box.delete(0, END)
    linear_drift_box.delete(0, END)

    global number_days
    number_days = entry_No_days_int.get()
    global recent_data
    recent_data = get_recent_frequencies(past_days = number_days)

    global plot1
    plot1 = fig.add_subplot(111)
    plot1.scatter(recent_data['timestamp'],recent_data['f0'],color='skyblue',label='Raw data')
    plot1.vlines(mjd_now(),-0.5e-14,1.5e-14,color='teal',label='Current MJD')
    plot1.set_title('Caesium Fountain data past '+str(number_days)+' days!')
    plot1.set_xlabel('MJD')
    plot1.set_ylabel('Fractional Frequency')


    global shifted_f0
    shifted_f0 = []
    global times
    times = []
    Time_CSF_array = np.array(recent_data['timestamp'])    
    F0_CSF_array = np.array(recent_data['f0']) 

    for i in np.arange(0,len(read_steer_history4),1) :
    
        if i!= len(read_steer_history4)-1:
        
            MJD1 = read_steer_history4[i,0]
            MJD2 = read_steer_history4[i+1,0]
            
            first_cut = recent_data[(recent_data['timestamp'])>MJD1]
            data_in_MJD1_MJD2_range = first_cut[(first_cut['timestamp'])<MJD2]
            f0_range = (np.array(data_in_MJD1_MJD2_range['f0']))

            shifted_array = f0_range - read_steer_history4[i,1]
            shifted_f0.extend(shifted_array)

            time_cut = np.array(data_in_MJD1_MJD2_range['timestamp'])
            times.extend(time_cut)    
        
        elif i == len(read_steer_history4)-1:
        
            MJD1 = read_steer_history4[i,0]
            
            end_cut = recent_data[(recent_data['timestamp'])>MJD1]
            f0_end_range = (np.array(end_cut['f0']))
        
        
            shifted_end = f0_end_range - read_steer_history4[-1,1]
            shifted_f0.extend(shifted_end)        
        
            time = np.array(end_cut['timestamp'])
            times.extend(time)

    plot1.plot(times,shifted_f0,'x',color='orange',label='Steered data')
        
    plot1.legend()
    canvas.draw_idle()
    fig.canvas.flush_events()

    window.after(600000,plot)


def save_data():

    os.remove(r'C:\Users\bethx\OneDrive\Desktop\Steered_CSF_data.csv')
    os.remove(r'C:\Users\bethx\OneDrive\Desktop\Raw_CSF_data.csv')

    shifted_data = {'MJD': times,
            'Steered Frequencies':shifted_f0}
    df = pd.DataFrame(shifted_data, columns=['MJD', 'Steered Frequencies'])

    recent_data.to_csv(r'C:\Users\bethx\OneDrive\Desktop\Raw_CSF_data.csv', index=True)
    df.to_csv(r'C:\Users\bethx\OneDrive\Desktop\Steered_CSF_data.csv', index=True)

    


def allan_deviations():

    fig.clear()
    plot1 = fig.add_subplot(111)

    (t_raw, ad_raw, ade_raw, adn_raw) = allantools.oadev(np.array(recent_data['f0']),1/3600, data_type="freq",taus=np.logspace(0,6,100))
    (t_shifted, ad_shifted, ade_shifted, adn_shifted) = allantools.oadev(shifted_f0,1/3600, data_type="freq",taus=np.logspace(0,6,100))


    plot1.loglog(t_raw,ad_raw,'o',color='skyblue',label='Raw data AD')
    plot1.loglog(t_shifted,ad_shifted,'x',color='orange',label='Steered data AD')
    plot1.set_title('Comparison of Allan Deviations')
    plot1.set_xlabel('Gate Time')
    plot1.set_ylabel('Overlapping Allan Deviation')
    plot1.grid()
    plot1.legend()

    canvas.draw_idle()
    fig.canvas.flush_events()
    


def timer(seconds):   
    timer_box.delete(0,END)
                                 
    if seconds > 0:
        timer_box.insert(0,seconds)
        window.after(1000,timer,seconds-1)

    elif seconds <= seconds - 10:
        
        window.after(1000,timer,seconds-1)                                                    


def read_in_file_and_store():
    read_data = np.loadtxt(r"C:\Users\bethx\OneDrive\Desktop\CSF_steer_history.txt")

    MJD_interest = read_data[-1,0]
    steer_period = read_data[-1,2]

    return MJD_interest, steer_period




def insert_f0():
        last_steer_box.delete(0,END)

        read_data = np.loadtxt(r"C:\Users\bethx\OneDrive\Desktop\CSF_steer_history.txt")

        f01 = read_data[-1,1]
        last_steer_box.insert(0,f01)

        window.after(500,insert_f0)



def write_to_LD_file():

    with open(r"C:\Users\bethx\OneDrive\Desktop\Fitted_linear_drifts.txt",'a') as LD_file:
        LD_file.write(str(mjd_now()))
        LD_file.write('    ')
        LD_file.write(linear_drift_box.get()+'\n')

        entry_LD_box.delete(0,END)
        entry_LD_box.insert(0,str(linear_drift_box.get()))


def save_plot():

    fig.clear()
    plot1 = fig.add_subplot(111)
    plot1.scatter(recent_data['timestamp'],recent_data['f0'],color='skyblue',label='Raw data')
    plot1.vlines(mjd_now(),-0.5e-14,1.5e-14,color='teal',label='Current MJD')
    plot1.set_title('Caesium Fountain data past '+str(number_days)+' days!')
    plot1.set_xlabel('MJD')
    plot1.set_ylabel('Fractional Frequency')
    plot1.plot(times,shifted_f0,'x',color='orange',label='Steered data')
    plot1.legend()

    canvas.draw_idle()
    fig.canvas.flush_events()



def fit_hours():

    fig.clear()

    plot1 = fig.add_subplot(111)
    plot1.scatter(recent_data['timestamp'],recent_data['f0'],color='skyblue',label='Raw data')
    plot1.vlines(mjd_now(),-0.5e-14,1.5e-14,color='teal',label='Current MJD')
    plot1.set_title('Caesium Fountain data past '+str(number_days)+' days!')
    plot1.set_xlabel('MJD')
    plot1.set_ylabel('Fractional Frequency')
    plot1.plot(times,shifted_f0,'x',color='orange',label='Steered data')
    plot1.legend()

    output_box.delete(0, END)
    linear_drift_box.delete(0, END)

    number_hours_fit = entry_fit_to_Y_days_int.get()/24
    
    if number_hours_fit > 0:

        data_for_fit = recent_data[recent_data['timestamp']>mjd_now()-number_hours_fit]
        m,b = np.polyfit(data_for_fit['timestamp'],data_for_fit['f0'],1)      
        plot1.plot(data_for_fit['timestamp'],m*(np.array(data_for_fit['timestamp']))+b,color='navy')

        linear_drift_box.insert(0,round(m,22))


        def line(x):
            return m*x + b

        extrapolate_to_now = round(line(mjd_now()),20)
        output_box.insert(0,extrapolate_to_now)
            
        last_pt = np.array(data_for_fit['timestamp'])[-1]

        extrapolate_range = np.linspace(last_pt,mjd_now(),100)
        plot1.plot(extrapolate_range,line(extrapolate_range),linestyle='dashed',color='navy')

    canvas.draw_idle()
    fig.canvas.flush_events()



def fit_days():

    fig.clear()

    plot1 = fig.add_subplot(111)
    plot1.scatter(recent_data['timestamp'],recent_data['f0'],color='skyblue',label='Raw data')
    plot1.vlines(mjd_now(),-0.5e-14,1.5e-14,color='teal',label='Current MJD')
    plot1.set_title('Caesium Fountain data past '+str(number_days)+' days!')
    plot1.set_xlabel('MJD')
    plot1.set_ylabel('Fractional Frequency')
    plot1.plot(times,shifted_f0,'x',color='orange',label='Steered data')
    plot1.legend()

    output_box.delete(0, END)
    linear_drift_box.delete(0, END)
    
    number_days_fit = entry_fit_to_Y_days_int.get()
    max_no_days = entry_No_days_int.get()
    
    if 0 < number_days_fit <= max_no_days:

        

        data_for_fit = recent_data[recent_data['timestamp']>mjd_now()-number_days_fit]
        m,b = np.polyfit(data_for_fit['timestamp'],data_for_fit['f0'],1)      
        plot1.plot(data_for_fit['timestamp'],m*(np.array(data_for_fit['timestamp']))+b,color='navy')

        linear_drift_box.insert(0,round(m,22))


        def line(x):
            return m*x + b

        extrapolate_to_now = round(line(mjd_now()),20)
        output_box.insert(0,extrapolate_to_now)
            
        last_pt = np.array(data_for_fit['timestamp'])[-1]

        extrapolate_range = np.linspace(last_pt,mjd_now(),100)
        plot1.plot(extrapolate_range,line(extrapolate_range),linestyle='dashed',color='navy')

    canvas.draw_idle()
    fig.canvas.flush_events()



def Switch_Fits(value):

    if value == "Hours":
        fit_hours()

    elif value == "Days":
        fit_days()



def steering_mode():

    timer_box.delete(0,END)
    time_of_steer,steering_period = read_in_file_and_store()
    
    time_of_next_steer = time_of_steer*24 + steering_period
    time_now = mjd_now()*24

    time_until_next_steer = round(time_of_next_steer - time_now,2)
    begin_countdown = round(time_until_next_steer*3600,2)

    timer(begin_countdown)
    insert_f0()

    def check_timer():
        time_now = mjd_now()*24

        if time_now < time_of_next_steer:
            print('yay')
            

        elif time_now >= time_of_next_steer:

            interpolate_LD()
            Switch_Fits(clicked1.get())

            f0 = output_box.get()
            steer_time = suggest_steer_box.get()

            last_steer_box.insert(0,f0)

            with open(r"C:\Users\bethx\OneDrive\Desktop\CSF_steer_history.txt", "a") as file:
                file.write(str(mjd_now()))
                file.write('    ')       
                file.write(str(f0))
                file.write('    ')
                file.write(str(steer_time)+'\n')

                window.after(100,steering_mode)

        window.after(300000,check_timer)

    check_timer()

        

def Switch_Plots(value):
    fig.clear()
    
    if value == "Plot recent data":
        save_plot()

    elif value == "Allan Deviations":
        allan_deviations()
        



#### Configure GUI ##############################################################

# RW and LD input 
input_label_RW = Label(left_frame, text="Random Walk of Maser",bg='azure')
input_label_LD = Label(left_frame, text="Linear Drift of Maser",bg='azure')

entry_RW_int = tk.DoubleVar()            
entry_RW_box = ttk.Entry(master=left_frame,
                              textvariable = entry_RW_int)
entry_RW_int.set(1.0)

entry_LD_int = tk.DoubleVar()            
entry_LD_box = ttk.Entry(master=left_frame,
                              textvariable = entry_LD_int)

read_LD_data = np.loadtxt(r"C:\Users\bethx\OneDrive\Desktop\Fitted_linear_drifts.txt")
Last_measured_LD= read_LD_data[-1,1]

entry_LD_int.set(Last_measured_LD)

Get_steer_button = ttk.Button(master=left_frame,
                            text='Get Steer Time',
                            command=interpolate_LD)


input_label_RW.grid(row=0,column=0)
entry_RW_box.grid(row=0,column=1,padx=10)
input_label_LD.grid(row=1,column=0)
entry_LD_box.grid(row=1,column=1,padx=10,pady=10)
Get_steer_button.grid(row=1,column=3,padx=10)


# Suggest steer output
suggest_steer_box = ttk.Entry(master= left_frame)
steer_label = Label(master=left_frame,
                         text='Implement steer every',
                          bg='azure')
steer_label2 = Label(master=left_frame,
                         text='hours :)'
                         ,bg='azure')

suggest_steer_box.insert(0,0)

steer_label.grid(row=2,column=0,padx=10)
suggest_steer_box.grid(row=2,column=1,pady=20)
steer_label2.grid(row=2,column=2,padx=10)




#  Entry box and button for X number days
input_label = Label(left_frame, text="Plot",bg='azure')
input_label2 = Label(left_frame, text="Days",bg='azure')

entry_No_days_int = tk.IntVar()           
entry_No_days_box = ttk.Entry(master=left_frame,
                              textvariable = entry_No_days_int)
entry_No_days_int.set(30)

No_days_button = ttk.Button(master=left_frame,
                            text='Plot',
                            command=plot)


input_label.grid(row=4,column=0)
entry_No_days_box.grid(row=4,column=1,pady=5)
input_label2.grid(row=4,column=2)
No_days_button.grid(row=4,column=3,padx=10)


# Canvas for plot
fig = Figure(dpi = 100)
canvas = FigureCanvasTkAgg(fig, master = right_frame)
canvas.get_tk_widget().grid(row=1,column=0,pady=5)
plot1 = fig.add_subplot(111)
plot1.set_title('Caesium Fountain data past N days!')
plot1.set_xlabel('MJD')
plot1.set_ylabel('Fractional Frequency')


## Option for Allan Deviation
plot_choices = [
    "Plot recent data",
    "Allan Deviations"
]
clicked = StringVar()
clicked.set( "Plot recent data" )
plot_choices_dropdown = OptionMenu(right_frame, clicked, *plot_choices, command = Switch_Plots)
plot_choices_dropdown.grid(row=0, column=0, sticky="W")


##Option for hours or days
fit_choices = ["Hours","Days"]
clicked1 = StringVar()
clicked1.set("Hours")
fit_choices_dropdown = OptionMenu(left_frame,clicked1,*fit_choices, command=Switch_Fits)
fit_choices_dropdown.grid(row=5,column=2)



#Fit line box and button
input_labelY = Label(left_frame, text="Fit",bg='azure')

entry_fit_to_Y_days_int = tk.IntVar()            
entry_fit_to_Y_days_box = ttk.Entry(master=left_frame,
                              textvariable = entry_fit_to_Y_days_int)

Yfit_button = ttk.Button(master=left_frame,
                            text='Fit',
                            command=lambda: Switch_Fits(clicked1.get()))

entry_fit_to_Y_hours_int = tk.IntVar()
entry_fit_to_Y_hours_box = ttk.Entry(master=left_frame,
                              textvariable = entry_fit_to_Y_hours_int)


input_labelY.grid(row=5,column=0)
entry_fit_to_Y_days_box.grid(row=5,column=1,pady=5)
Yfit_button.grid(row=5,column=3)



#F0 output 
output_box = ttk.Entry(master= left_frame)
output_label = Label(master=left_frame,
                         text='F offset =',
                         bg='azure')

output_label.grid(row=7,column=0)
output_box.grid(row=7,column=1,padx=10,pady=15)


## Linear drift output and button

linear_drift_box = ttk.Entry(master= left_frame)
linear_drift_box_label = Label(master=left_frame,
                         text=' Linear Drift =',
                         bg='azure')

save_LD_to_file = ttk.Button(master=left_frame,
                             text = 'Save to File',
                             command = write_to_LD_file)


save_LD_to_file.grid(row=8,column=3)
linear_drift_box_label.grid(row=8,column=0)
linear_drift_box.grid(row=8,column=1,padx=10,pady=5)


## Empty label for formatting

empty_label = Label(master=left_frame,text='         ',bg='azure')
empty_label.grid(row=3,column=0,pady=15)


## Export data to file

export_to_file = ttk.Button(master=left_frame,
                             text = 'Export CSF data to File',
                             command = save_data)


export_to_file.grid(row=6,column=3,pady=15)


## Add Initials

Name_label = Label(right_frame, text="Developed by Beth :)",bg='azure',fg='darkgrey')
Name_label.grid(row=0, column=0, sticky="E")


# STEERING MODE


# Last steer

last_steer_box= ttk.Entry(master= steering_mode_frame,width=15)
last_steer_label = Label(master=steering_mode_frame,
                         text='Last Steer')

last_steer_label.grid(row= 0,column=0,padx=20)
last_steer_box.grid(row= 0,column=1,padx=10)


# Timer Box

timer_box= ttk.Entry(master= steering_mode_frame,width=15)
timer_label = Label(master=steering_mode_frame,
                         text='Countdown to next steer')

timer_label.grid(row= 0,column=2,padx=20)
timer_box.grid(row= 0,column=3,padx=10,pady=20)


## Check steer Button

#check_steer_button = ttk.Button(master=steering_mode_frame, text="Steering Mode!!", command=steering_mode)
#check_steer_button.grid(row=0,column=4,padx=10)


# Exit Button
button1 = ttk.Button(master=steering_mode_frame, text="Exit", command=window.destroy)
button1.grid(row= 1,column=3,pady=10)

plot()
interpolate_LD()
Switch_Fits(clicked1.get())
steering_mode()


window.mainloop()


