import numpy as np
import pandas as pd
import imufusion
from scipy.signal import butter, filtfilt

#order:
# butterworth --> madgwick --> remove gravity

#butterworth normal for IMU, exact values changable in app.py
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # Check for valid cutoff
    if normal_cutoff >= 1.0:
        return data 
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Main
def process_imu_data(df, enable_butterworth=False, cutoff_freq=8.0, sensors='both'):
   
    # Create a copy to avoid SettingWithCopy warnings
    proc_df = df.copy()
   #put timestamp in s
    timestamps_sec = proc_df['timestamp'].to_numpy() / 1000.0
    dt = np.diff(timestamps_sec, prepend=timestamps_sec[0])
    # Fix first sample or 0s jic
    median_dt = np.median(dt[dt > 0.0001])
    dt[dt <= 0.0001] = median_dt
    #calc sampling frequency
    avg_dt = np.mean(dt)
    sample_rate_est = 1.0 / avg_dt if avg_dt > 0 else 100.0
    

   #Butterworth
    if enable_butterworth:
        # Define the raw columns we want to smooth
        raw_cols = [
            'acc_x_chest', 'acc_y_chest', 'acc_z_chest'
        ]
        
        for col in raw_cols:
            if col in proc_df.columns:
                # Interpolate small gaps if any, to prevent filter crash
                clean_col = proc_df[col].interpolate().bfill().ffill()
                proc_df[col] = butter_lowpass_filter(clean_col.values, cutoff_freq, sample_rate_est)
        

    # Madgwick, 0.5 gain normal for human movement and this was settings in Thomas old code
    settings = imufusion.Settings(
        imufusion.CONVENTION_NWU,
        0.5,   # gain
        2000,  # gyro range (deg/s)
        10,    # accel rejection (g)
        10,    # mag rejection (uT)
        int(5 * sample_rate_est)
    )
    #only apply to chet data unless because don't want madgwick on high acceleration wrist data, waste of processing for real-time
    locs_to_process = ['chest'] if sensors == 'chest_only' else ['chest', 'wrist']

    for loc in locs_to_process:
        if f'acc_x_{loc}' not in proc_df.columns:
            continue

        #prep arays:
        gyro = proc_df[[f'gyro_x_{loc}', f'gyro_y_{loc}', f'gyro_z_{loc}']].to_numpy()
        # convert m/s2 to g
        accel = proc_df[[f'acc_x_{loc}', f'acc_y_{loc}', f'acc_z_{loc}']].to_numpy() / 9.81
        mag = proc_df[[f'magn_x_{loc}', f'magn_y_{loc}', f'magn_z_{loc}']].to_numpy()

        # Madgwick fusion
        offset = imufusion.Offset(int(sample_rate_est))
        ahrs = imufusion.Ahrs()
        ahrs.settings = settings

        n = len(proc_df)
        euler = np.empty((n, 3))
        accel_global = np.empty((n, 3))
        linear_accel_z = np.empty(n) # Vertical acceleration without gravity for integration for min/max velocity

# Madwick stuff:
        for i in range(n):
            gyro[i] = offset.update(gyro[i]) # Remove gyro bias
            ahrs.update(gyro[i], accel[i], mag[i], dt[i])

            R = ahrs.quaternion.to_matrix() #rotate to global frame
            
            # accel_global = Gravity (1g) + User Motion
            accel_global[i] = R @ accel[i] 
            
            # Save Euler angles (Roll, Pitch, Yaw)
            euler[i] = ahrs.quaternion.to_euler()
            
            #put back in m/s^2 now thta madgwick is done
            linear_accel_z[i] = (accel_global[i, 2] - 1.0) * 9.81

        #save euler angles and everything else to data frame but we dont use them later
        proc_df[f'roll_{loc}'] = euler[:, 0]
        proc_df[f'pitch_{loc}'] = euler[:, 1]
        proc_df[f'yaw_{loc}'] = euler[:, 2]
        
        proc_df[f'acc_x_global_{loc}'] = accel_global[:, 0]
        proc_df[f'acc_y_global_{loc}'] = accel_global[:, 1]
        proc_df[f'acc_z_global_{loc}'] = accel_global[:, 2]
        
        proc_df[f'lin_acc_z_{loc}'] = linear_accel_z

    return proc_df