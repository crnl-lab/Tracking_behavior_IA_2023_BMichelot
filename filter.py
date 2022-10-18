"""
Smoothing script for OpenPose
Author: Alexandre Bremard | INSA-Lyon
GitHub: TODO
"""

import sys
import cv2
import numpy as np
import time
import statistics
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import savgol_filter
    
# apply filter
def savgol(source, window_size, polynome_degree, save_output= True, time_serie= None):
    """
        Wrapper of the scipy.signal.savgol_filter for OpenPose

        Args:
            source (str): data path on machine, must be in .npy format
            window_size (int): window size as defined in the scipy documentation
            threshold (float): polynome degree for regression as defined in the scipy documentation, has to be lower than window_size
        Returns:
            np.array: always returns filtered array            
    """
    if time_serie is None:
        # Parse variables
        filename = source.split('.')[0]
        time_serie = np.load(source)
    filtered_time_serie = np.array(time_serie, copy=True)

    if time_serie.shape[2] >= window_size:
        # Apply filter
        for kid, keypoint in enumerate(filtered_time_serie):
            for cid, coord in enumerate(keypoint):
                if cid != 2:
                    filtered_time_serie[kid][cid] = savgol_filter(time_serie[kid][cid], window_size, polynome_degree)
        
        if save_output:
            np.save('{}_filtered.npy'.format(filename), filtered_time_serie)

    else:
        print(f"Time serie size must be at least equal to window_size, returning unfiltered array")

    return filtered_time_serie

def moving_average(source, window_size, save_output= True, time_serie= None):
    """
        Moving average filter

        Args:
            source (str): data path on machine, must be in .npy format
            window_size (int): window size used to compute mean
        Returns:
            np.array: always returns filtered array            
    """

    if time_serie is None:
        # Parse variables
        filename = source.split('.')[0]
        time_serie = np.load(source)

    # Apply filter
    filtered_time_serie = np.array(time_serie, copy=True)
    for kid, keypoint in enumerate(filtered_time_serie):
        for cid, coord in enumerate(keypoint):
            if cid != 2:
                filtered_time_serie[kid][cid][window_size-1:] = pd.DataFrame(time_serie[kid][cid]).rolling(window_size).mean()[window_size-1:].iloc[:,0]

    if save_output:
        np.save('{}_filtered.npy'.format(filename), filtered_time_serie)

    return filtered_time_serie

def moving_median(source, window_size, save_output= True, time_serie= None):
    """
        Moving median filter

        Args:
            source (str): data path on machine, must be in .npy format
            window_size (int): window size used to compute median
        Returns:
            np.array: always returns filtered array            
    """

    if time_serie is None:
        # Parse variables
        filename = source.split('.')[0]
        time_serie = np.load(source)

    # Apply filter
    filtered_time_serie = np.array(time_serie, copy=True)
    for kid, keypoint in enumerate(filtered_time_serie):
        for cid, coord in enumerate(keypoint):
            if cid != 2:
                filtered_time_serie[kid][cid][window_size-1:] = pd.DataFrame(time_serie[kid][cid]).rolling(window_size).median()[window_size-1:].iloc[:,0]

    if save_output:
        np.save('{}_filtered.npy'.format(filename), filtered_time_serie)

    return filtered_time_serie    

def flat_median_filter(source:str, window_size:int, threshold:float= 5.0, flat_window:float= None, flat_tolerance:float= 0.1, save_output:bool= True, raw:bool= None) -> np.array:
    """
        2-step filter:
        1) compute median and update data only when variations are above threshold
        2) compute stability and clip data whenever it becomes stable 
    Args:
        source (str): data path on machine, must be in .npy format
        window_size (int): window size used to compute median
        threshold (float, optional): threshold for median variation. Defaults to 5.0.
        flat_window (float, optional): window size used to clip data when stable. Defaults to None.
        flat_tolerance (float, optional): stability tolerance. Defaults to 0.1.
        save_output (bool, optional): save output as '{raw_file}_filtered.npy'. Defaults to True.

    Returns:
        np.array: always returns filtered array
    """        

    # default flat_window value is window_size
    if flat_window is None:
        flat_window = window_size

    # Parse variables
    if raw is None:
        # Parse variables
        filename = source.split('.')[0]
        raw = np.load(source)
    
    # Filtered data is initialized as a deep copy of raw data
    filtered = np.array(raw, copy=True)

    for kid, keypoint in enumerate(raw):
        for cid, coord in enumerate(keypoint):
            if cid != 2:
                # Previous median is the window's first value
                prev_med = coord[0]
                for tid, time_unit in enumerate(coord):
                    window_slice = coord[tid-window_size:tid]
                    # Filtering only after window_size-th frame
                    if len(window_slice) >= window_size:
                        med = statistics.median(window_slice)
                        if abs(med - prev_med) > threshold:
                            """
                                The 1st layer of filter will update the data only when the window median's variations are above the threshold
                            """                        
                            filtered[kid][cid][tid] = statistics.median(window_slice)
                            prev_med = med
                        elif abs(np.mean(raw[kid][cid][tid:tid+flat_window]) - raw[kid][cid][tid]) < flat_tolerance:
                            """
                                When the median becomes stable after a spike, the data can sometimes be stuck close to the actual values,
                                since there are no major variations in the median
                                The 2nd layer of filter will clip the data to the actual values whenever the data becomes stable after a spike
                            """                            
                            filtered[kid][cid][tid] = raw[kid][cid][tid]
                        else:
                            filtered[kid][cid][tid] = filtered[kid][cid][tid-1]

    if save_output:
        np.save(f'{filename}_filtered.npy', filtered)

    return filtered

if __name__ == "__main__":

    source = sys.argv[1] # 'media/test_patient.npy'
    window_size = int(sys.argv[2])
    polynome_degree = int(sys.argv[3])
    threshold = float(sys.argv[4])

    # filtered_time_serie = savgol(source, window_size, polynome_degree)
    # filtered_time_serie = moving_average(source, window_size)
    # filtered_time_serie = flat_median_filter(source, window_size, threshold)
    # filtered_time_serie = moving_median(source, window_size)