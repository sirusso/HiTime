import pandas as pd
import numpy as np


# parameters may need to be adjusted for different people but this worked best for these three
# Added take off and landing force check to get rid of false positives from calf raises
#without the velocity based detection the results are just not good

#FLIGHT TIME BASED in global frame





# Set up
G_CONST = 9.81  

def detect_jumps(df, flight_threshold=0.5, min_takeoff_vel=1.0, min_takeoff_accel_g=0.0, min_landing_accel_g=0.0, min_height_cm=10.0, max_height_cm=120.0):
    """
    Analyzes processed IMU data to find jumps based on adjustable parameters.
    """
    #get madgwick fitlered chest data
    if df.empty or 'acc_z_global_chest' not in df.columns:
        return []
        
    time_s = df['timestamp'].values / 1000.0
    dt = np.mean(np.diff(time_s))
    if dt == 0: return []
    freq = 1.0 / dt
    
    acc_raw_g = df['acc_z_global_chest'].values
    
    #subtract gravity before integration
    if 'lin_acc_z_chest' in df.columns:
        acc_dynamic_ms2 = df['lin_acc_z_chest'].values
    else:
        acc_dynamic_ms2 = (acc_raw_g - 1.0) * G_CONST #its in the file though so shouldnt have problem
    
    #if global z acc is below threshold then possibly in the air
    is_flight_candidate = acc_raw_g < flight_threshold
    
    flight_epochs = []  #init. list to hold start and end indices of flight phases
    in_flight = False
    start_idx = 0
    
    # add flight epochs to list but ignore if its shorter than 0.1s bc noise
    for i in range(len(is_flight_candidate)):
        if is_flight_candidate[i] and not in_flight:
            in_flight = True
            start_idx = i
        elif not is_flight_candidate[i] and in_flight:
            in_flight = False
            if (time_s[i] - time_s[start_idx]) > 0.10: 
                flight_epochs.append((start_idx, i))

    # merge epochs that are close together otherwise it messes stuff up
    merged_epochs = []
    if flight_epochs:
        curr_start, curr_end = flight_epochs[0]
        for next_start, next_end in flight_epochs[1:]:
            if (time_s[next_start] - time_s[curr_end]) < 0.2:
                curr_end = next_end
            else:
                merged_epochs.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged_epochs.append((curr_start, curr_end))
    
    # CHECKS BASED ON SLIDER SETTINGS!!! - can be adjusted in streamlit app
    detected_jumps = []
    for (f_start, f_end) in merged_epochs:
        
        # Takeoff push check: check 0.3s (changed from 0.5s 4/12) before jump to check for strong push off acceleration to help get rid of false postitives from running or squatting
        push_start = max(0, f_start - int(0.3 * freq))
        pre_flight_accel = acc_raw_g[push_start:f_start]
        max_accel_push = np.max(pre_flight_accel) if len(pre_flight_accel) > 0 else 0
        
        if max_accel_push < min_takeoff_accel_g:
            continue
            
        # Landing force check: check 0.3s after jump for impact w ground to help w running and squating fp's
        impact_end = min(len(acc_raw_g), f_end + int(0.3 * freq))
        post_flight_accel = acc_raw_g[f_end:impact_end]
        max_accel_impact = np.max(post_flight_accel) if len(post_flight_accel) > 0 else 0
        
        if max_accel_impact < min_landing_accel_g:
            continue
            
        # Continue with normal velocity/height math
        #buffers to get more of approach and landing
        win_start = max(0, f_start - int(1.5 * freq))
        win_end = min(len(acc_raw_g), f_end + int(1.0 * freq))
        
        chunk_time = time_s[win_start:win_end]

        if len(chunk_time) < 10: continue #was giving some super high jumps before so needed to get rid of those

        # Cumsum only over this window so drift wont be a problem
        chunk_vel = np.cumsum(acc_dynamic_ms2[win_start:win_end]) * dt
        
        rel_f_start = f_start - win_start #relative index of fligt start in this current bit of data
        search_limit = min(len(chunk_vel), rel_f_start + int(0.5 * freq))
        
        takeoff_rel_idx = np.argmax(chunk_vel[:search_limit])   #max velocity is takeoff
        takeoff_val = chunk_vel[takeoff_rel_idx]
        
        landing_search_slice = chunk_vel[takeoff_rel_idx:]
        if len(landing_search_slice) == 0: continue
        
        landing_rel_idx = takeoff_rel_idx + np.argmin(landing_search_slice) #lowest velocity (negative) is at landing
        
        #now get flight time
        t_takeoff = chunk_time[takeoff_rel_idx]
        t_landing = chunk_time[landing_rel_idx]
        ft = t_landing - t_takeoff
        

        if ft < 0.2 or ft > 1.5: continue  #wide checks, but have sliders to set more realistic height thresholds as a parameter 
        if takeoff_val < min_takeoff_vel: continue     
        
        # height from flight time calc
        h_raw_m = (G_CONST * ft**2) / 8.0
        h_cm = h_raw_m * 100.0
        
        #check that hieght is within user set desired range
        if min_height_cm <= h_cm <= max_height_cm:
            detected_jumps.append({
                "takeoff_t": t_takeoff,
                "landing_t": t_landing,
                "flight_time_s": ft,
                "height_cm": h_cm,
                "takeoff_vel": takeoff_val
            })

# Before we discarded a jump if it was within 0.5s of another jump which was making some takeoff points be set way to early, so now we will try taking the one w higher takeoff velocity
    
    # Sort by takeoff time to process in order
    detected_jumps.sort(key=lambda x: x['takeoff_t'])
    
    unique_jumps = []
    if detected_jumps:
        current_best = detected_jumps[0]
        
        for next_jump in detected_jumps[1:]:
            # If the next jump is within 0.5s of the current candidate, they are duplicates
            if next_jump['takeoff_t'] - current_best['takeoff_t'] < 0.5:
                
                # Compare takeoff velocity, not calculated height!
                if next_jump['takeoff_vel'] > current_best['takeoff_vel']:
                    current_best = next_jump
            else:
                unique_jumps.append(current_best)
                current_best = next_jump
        
        unique_jumps.append(current_best)

    POST_LANDING_BUFFER = 0.3
    filtered_jumps = []
    for jump in unique_jumps:
        if filtered_jumps:
            if jump['takeoff_t'] - filtered_jumps[-1]['landing_t'] < POST_LANDING_BUFFER:
                continue
        filtered_jumps.append(jump)

    return filtered_jumps