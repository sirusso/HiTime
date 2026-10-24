import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

#This only checks the highest peak and if it deosn't fit the threshold criteria it moves on to the next jump,
# if this is causing a lot of misses then I coud change it to check the next few highest peaks and see if 
# they fulfill the threshold criteria

# Set up
HP_CUTOFF = 25.0       # i think 30 is too high, maybe 20 ok   
WINDOW_MS = 30           
VERIFY_WIDTH_MS = 30     #check jerk and gyro arround accel peak in this window, jerk peak comes slightly after accel peak
G_CONST = 9.81           # for converting m/s2 to g

def compute_hit_signals(df):
    if df.empty or 'acc_x_wrist' not in df.columns: #just incase theres no data
        df['Wrist_Mag_g'] = 0.0
        df['Hit_Energy_Envelope'] = 0.0
        return df

    # Convert to seconds and calc sampling frequency
    time_s = df['timestamp'].values / 1000.0
    dt = np.mean(np.diff(time_s))
    if dt == 0: 
        df['Wrist_Mag_g'] = 0.0
        df['Hit_Energy_Envelope'] = 0.0
        return df
        
    fs = 1.0 / dt
    
    ax, ay, az = df['acc_x_wrist'].values, df['acc_y_wrist'].values, df['acc_z_wrist'].values
    
    #in m/s^2
    raw_mag_ms2 = np.sqrt(ax**2 + ay**2 + az**2)
    
    # High Pass filter to get rid of arm swing (I hope)
    nyq = 0.5 * fs
    normal_cutoff = HP_CUTOFF / nyq
    
    # 4th order Butterworth is standard
    if normal_cutoff < 1.0:
        b, a = butter(4, normal_cutoff, btype='high')
        sig_highpass = filtfilt(b, a, raw_mag_ms2)
    else:
        sig_highpass = raw_mag_ms2
        
    # Convert high-passed m/s^2 back to g since threshold is in g
    df['Wrist_Mag_g'] = sig_highpass / G_CONST 
    
    # Calculate jerk energy envelope - based on Mariah saying to look into Pan Tompkins
    window_samples = max(1, int((WINDOW_MS / 1000.0) * fs))
    
    # derivative using np dif and not gradient since smapling frequency is technically not high enough to capture the hit
    jerk = np.diff(sig_highpass, prepend=sig_highpass[0]) * fs
    
    # Square to emphasize signnal
    jerk_sq = jerk**2
    
    # Moving Integration (Envelope)
    #envelope = pd.Series(jerk_sq).rolling(window_samples, center=True).mean().fillna(0).values

    # Replacing moving integration with convolution because it is supposedly the same after the first few samples of the window and faster
    kernel = np.ones(window_samples) / window_samples
    envelope = np.convolve(jerk_sq, kernel, mode='same')
    
    df['Hit_Energy_Envelope'] = envelope
    
    return df

def detect_hits(df, jumps, min_energy=500.0, min_gyro=300.0, min_accel_g=2.0):
    """
    Analyzes processed IMU data to find volleyball hits within detected jump windows.
    """
    if df.empty or not jumps:
        return jumps
    
    df = compute_hit_signals(df)
    
    # Extract arrays
    time_s = df['timestamp'].values / 1000.0
    filtered_mag_g = df['Wrist_Mag_g'].values
    envelope = df['Hit_Energy_Envelope'].values
    
    gx, gy, gz = df['gyro_x_wrist'].values, df['gyro_y_wrist'].values, df['gyro_z_wrist'].values
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    
    #jump windows
    for jump in jumps:
        t_start = jump['takeoff_t']
        t_end = jump['landing_t']
        
        mask = (time_s >= t_start) & (time_s <= t_end)
        idx_in_window = np.where(mask)[0]
        
        if len(idx_in_window) == 0:
            jump['hit_detected'] = False
            continue
            
        #find highest acc peak in filtered data
        window_acc_g = filtered_mag_g[idx_in_window]
        local_peak_idx = np.argmax(window_acc_g)
        global_peak_idx = idx_in_window[local_peak_idx]
        t_candidate = time_s[global_peak_idx]
        peak_val_g = window_acc_g[local_peak_idx]
        
        #jerk peak
        window_energy = envelope[idx_in_window]
        local_jerk_idx = np.argmax(window_energy)
        global_jerk_idx = idx_in_window[local_jerk_idx]
        t_candidate_jerk = time_s[global_jerk_idx]
        
        #threshold check in short surrounding window
        verify_mask = (time_s >= t_candidate - (VERIFY_WIDTH_MS / 1000.0)) & \
                      (time_s <= t_candidate + (VERIFY_WIDTH_MS / 1000.0))
        verify_idx = np.where(verify_mask)[0]
        
        if len(verify_idx) > 0:
            check_energy = np.max(envelope[verify_idx])
            check_gyro = np.max(gyro_mag[verify_idx])
            
            is_valid = (check_energy > min_energy) and (check_gyro > min_gyro) and (peak_val_g >= min_accel_g)
        else:
            is_valid = False
            check_energy = 0.0
            check_gyro = 0.0
            
        jump['hit_detected'] = is_valid
        if is_valid:
            jump['hit_time'] = t_candidate
            jump['hit_time_jerk'] = t_candidate_jerk
            jump['hit_peak_g'] = peak_val_g
            jump['hit_energy'] = check_energy
            jump['hit_max_gyro'] = check_gyro
        else:
            jump['hit_time'] = np.nan
            jump['hit_time_jerk'] = np.nan
            jump['hit_peak_g'] = np.nan
            jump['hit_energy'] = np.nan
            jump['hit_max_gyro'] = np.nan

    return jumps