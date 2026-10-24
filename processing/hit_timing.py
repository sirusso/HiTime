G_CONST = 9.81  

#read me: thresholds for early/late set in the app
# if a player was late that means the contact is before the apex
# early means after apex

def evaluate_hit_timing(jumps, early_pct_threshold, late_pct_threshold):
    for jump in jumps:
        # Skip if no hit was detected for this jump
        if not jump.get('hit_detected') or 'hit_time' not in jump:
            jump['height_at_hit_cm'] = None
            jump['hit_height_pct'] = None
            jump['hit_timing'] = "No Hit"
            continue
            
        t_takeoff = jump['takeoff_t']
        t_landing = jump['landing_t']
        t_hit = jump['hit_time']
        
        # Max height of hit
        max_height_cm = jump.get('height_cm', 0.0) 
        
        # flight time and kinematics info
        ft = t_landing - t_takeoff
        v0 = (G_CONST * ft) / 2.0 # v0 derived from flight time, not sensor data since that didn't go well
        t_delta = t_hit - t_takeoff            
        
        # skip if something weird happened and hit is outside jump window
        if t_delta < 0 or t_delta > ft:
            jump['height_at_hit_cm'] = None # 4-15 there was nothing here before, added it to hopefully avoid dropping columns with no data
            jump['hit_height_pct'] = None
            jump['hit_timing'] = "Out of Bounds"
            continue
            
        #  h = v0*t - 0.5*g*t^2 to get height at timestamp of hit
        h_hit_m = (v0 * t_delta) - (0.5 * G_CONST * (t_delta ** 2))
        h_hit_cm = max(0, h_hit_m * 100.0) 
        
        # what percentag of the max was this?
        hit_height_pct = (h_hit_cm / max_height_cm * 100.0) if max_height_cm > 0 else 0.0
        
        jump['height_at_hit_cm'] = round(h_hit_cm, 1)
        jump['hit_height_pct'] = round(hit_height_pct, 1)
        
        # get the time of the apex
        t_apex = t_takeoff + (ft / 2.0)
        
        #check if hit was before or after apex
        if t_hit < t_apex:
            # hit on the way up / before apex
            # = player was late
            if hit_height_pct < late_pct_threshold:
                jump['hit_timing'] = "Late"
            else:
                jump['hit_timing'] = "On Time"
        else:
            # hit on the way down / after apex
            # = player jumped early
            if hit_height_pct < early_pct_threshold:
                jump['hit_timing'] = "Early"
            else:
                jump['hit_timing'] = "On Time"
                
    return jumps