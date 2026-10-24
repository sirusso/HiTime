import streamlit as st
import pandas as pd
import os
import time
import subprocess
import sys
import signal
import threading
import numpy as np
import altair as alt

# make sure all the imports from other files are there
try:
    from processing.synchronization import align_sensors_zeroed
    from processing.filters import process_imu_data
    from processing.jump_detection import detect_jumps
    from processing.hit_detection import detect_hits, compute_hit_signals
    from processing.hit_timing import evaluate_hit_timing
except ImportError:
    st.error("⚠️ Module not found. Check your 'processing' folder structure.")
    st.stop()

try:
    from realtime_worker import realtime_worker
except ImportError:
    st.error("⚠️ Module 'realtime_worker' not found. Check your project structure.")
    st.stop()

#config
DATA_FOLDER = "data_live"
SESSIONS_FOLDER = os.path.join(DATA_FOLDER, "sessions")
os.makedirs(SESSIONS_FOLDER, exist_ok=True)
CHEST_FILE = os.path.join(DATA_FOLDER, "chest.csv")
WRIST_FILE = os.path.join(DATA_FOLDER, "wrist.csv")
RECORDER_SCRIPT = "recorder.py"
RECORDER_LOG = os.path.join(DATA_FOLDER, "recorder.log")


def get_recorder_status():
    if not os.path.exists(RECORDER_LOG):
        return 'connecting', 'Starting up...'

    try:
        with open(RECORDER_LOG, 'r') as f:
            lines = f.readlines()

        for line in reversed(lines):
            line = line.strip()
            if line.startswith('STATUS: ERROR'):
                return 'error', line.replace('STATUS: ERROR - ', '')
            if line.startswith('STATUS: STREAMING'):
                return 'streaming', 'Both sensors connected & streaming'

        last = next((l.strip() for l in reversed(lines) if l.strip()), 'Starting up...')
        return 'connecting', last

    except Exception:
        return 'connecting', 'Starting up...'

# real-time window variables!
RT_WINDOW_SECONDS = 15
RT_WARMUP_SECONDS = 20 #this can maybe be shorter
SENSOR_HZ = 104 #sample rate, make sure to change this is sf changed!!

#old session files
def get_session_files(session_id):
    processed = os.path.join(SESSIONS_FOLDER, f"processed_session_{session_id}.csv")
    jumps = os.path.join(SESSIONS_FOLDER, f"detected_jumps_{session_id}.csv")
    rt_jumps = os.path.join(SESSIONS_FOLDER, f"rt_jumps_{session_id}.csv")
    return processed, jumps, rt_jumps


def list_sessions():
    files = [f for f in os.listdir(SESSIONS_FOLDER) if f.startswith("processed_session_")]
    ids = [f.replace("processed_session_", "").replace(".csv", "") for f in files]
    return sorted(ids, reverse=True)



# check if sensors are recording, didnt do anything with this though so app doesnt say anything if the recording stops
def is_data_fresh():
    if not os.path.exists(CHEST_FILE) or not os.path.exists(WRIST_FILE):
        return False
    try:
        return (time.time() - os.path.getmtime(CHEST_FILE)) < 2.0
    except Exception:
        return False

#timing symbols
def timing_emoji(timing):
    return {'On Time': '✅', 'Early': '⏫', 'Late': '⏬', 'No Hit': '', 'Out of Bounds': '⚠️'}.get(timing, '')

#visual jump parabola
def make_parabola_chart(jump: dict):
    height_cm = jump.get('height_cm', 0)
    ft = jump.get('flight_time_s', 0)

    if ft <= 0 or height_cm <= 0:
        return None

    G = 9.81
    v0 = (G * ft) / 2.0

    t_vals = np.linspace(0, ft, 40)
    h_vals = np.maximum(0, (v0 * t_vals - 0.5 * G * t_vals ** 2) * 100)  # metres -> cm

    curve_df = pd.DataFrame({'Time (s)': t_vals, 'Height (cm)': h_vals})

    y_max = height_cm * 1.35

    base = alt.Chart(curve_df).mark_line(
        color='#4c78a8', strokeWidth=2.5, clip=True
    ).encode(
        x=alt.X('Time (s):Q',
                axis=alt.Axis(format='.2f', tickCount=5, labelFontSize=9, titleFontSize=9),
                scale=alt.Scale(domain=[0, ft])),
        y=alt.Y('Height (cm):Q',
                scale=alt.Scale(domain=[0, y_max]),
                axis=alt.Axis(labelFontSize=9, titleFontSize=9, grid=False))
    )

    apex_df = pd.DataFrame({
        'Time (s)': [ft / 2.0],
        'Height (cm)': [height_cm],
        'Label': [f'Peak: {height_cm:.1f} cm']
    })
    apex_mark = alt.Chart(apex_df).mark_point(
        color='#4c78a8', size=90, filled=True
    ).encode(
        x='Time (s):Q',
        y='Height (cm):Q',
        tooltip=[alt.Tooltip('Label:N', title='')]
    )

    layers = [base, apex_mark]

    hit_detected = jump.get('hit_detected', False)
    hit_height = jump.get('height_at_hit_cm')
    hit_timing = jump.get('hit_timing', '')

    if hit_detected and hit_height is not None:
        t_takeoff = jump.get('takeoff_t', 0)
        t_hit_abs = jump.get('hit_time')

        if t_hit_abs is not None:
            t_hit = float(t_hit_abs) - float(t_takeoff)
            t_hit = max(0.0, min(t_hit, ft))
        else:
            t_hit = ft / 2.0

        color_map = {
            'On Time': '#2ca02c',
            'Early':   '#ff7f0e',
            'Late':    '#d62728',
        }
        hit_color = color_map.get(hit_timing, '#9467bd')

        label = f'{timing_emoji(hit_timing)} {hit_timing} — {hit_height:.1f} cm'

        hit_df = pd.DataFrame({
            'Time (s)': [t_hit],
            'Height (cm)': [float(hit_height)],
            'Label': [label]
        })

        hit_rule = alt.Chart(hit_df).mark_rule(
            color=hit_color, strokeDash=[4, 4], strokeWidth=1.5, opacity=0.7
        ).encode(x='Time (s):Q')

        # vball emoji at hit point
        hit_mark = alt.Chart(hit_df).mark_text(
            text='🏐',
            fontSize=22,
            dy=-14,
            clip=False
        ).encode(
            x='Time (s):Q',
            y='Height (cm):Q',
            tooltip=[alt.Tooltip('Label:N', title='Hit')]
        )

        layers += [hit_rule, hit_mark]

    chart = alt.layer(*layers).properties(height=160)

    return chart


for key, default in [
    ('recording_pid', None),
    ('current_session_id', None),
    ('selected_session_id', None),
    ('rt_shared', {}),
    ('rt_thread', None),
    ('live_jump_history', []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# UI LAYOUT
st.set_page_config(page_title="HiTime", layout="centered")
st.title("🏐 Jump Height + Hit Detection 🏐")
st.markdown("---")

# Detection settings, so can change defaults here if needed
with st.expander("⚙️ Detection Settings", expanded=False):
    st.caption("Choose Parameters for Jumps (Chest) and Hits (Wrist)")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.markdown("**Min jump thresholds**")
        flight_thresh = st.slider("Flight Threshold (g)", 0.2, 0.9, 0.65, 0.05)
        min_vel = st.slider("Min Takeoff Vel (m/s)", 0.1, 3.0, 2.3, 0.1,
                            help="If false jumps are detected, try raising this.")
        min_height = st.slider("Min Height (cm)", 5.0, 30.0, 10.0, 1.0)

    with col_s2:
        st.markdown("**More jump thresholds**")
        min_push = st.slider("Min Takeoff Push (g)", 0.0, 4.0, 2.3, 0.1,
                             help="lower if jumps are missed")
        min_land = st.slider("Min Landing Force (g)", 0.0, 5.0, 1.0, 0.1,
                             help="If false jumps are detected when you go on tippy toes, try raising this.")
        max_height = st.slider("Max Height (cm)", 50.0, 150.0, 90.0, 5.0)

    with col_s3:
        st.markdown("**Hit Detection**")
        min_accel = st.slider("Min Hit Accel (g)", 0.0, 5.0, 0.5, 0.05)
        min_energy_millions = st.slider("Min Jerk (Millions)", 0.0, 10.0, 1.0, 0.25,
                                        help="possibly helps exclude non hits or certain shots")
        min_gyro = st.slider("Min Gyro (deg/s)", 0.0, 1500.0, 300.0, 50.0)

    with col_s4:
        st.markdown("Hit timing")
        early_thresh = st.slider("Early Thresh (%)", 50, 100, 85, 1,
                                 help="If hitting on the way down and below this pct. of max jump height, mark Early.")
        late_thresh = st.slider("Late Thresh (%)", 50, 100, 90, 1,
                                help="If hitting on the way UP and below this pct. of max jump height, mark Late.")

st.markdown("---")


rt_params = {
    'jump': dict(
        flight_threshold=flight_thresh,
        min_takeoff_vel=min_vel,
        min_takeoff_accel_g=min_push,
        min_landing_accel_g=min_land,
        min_height_cm=min_height,
        max_height_cm=max_height,
    ),
    'hit': dict(
        min_energy=min_energy_millions * 1_000_000.0,
        min_gyro=min_gyro,
        min_accel_g=min_accel,
    ),
    'timing': dict(
        early_pct_threshold=early_thresh,
        late_pct_threshold=late_thresh,
    ),
    'rt_jumps_file': None,  # Populated at session start
}

#start/stop recording buttons
col1, col2 = st.columns(2)

with col1:
    if st.session_state.recording_pid is None:
        st.info("⚪ System Idle")
        sessions = list_sessions()
        if sessions:
            st.caption(f"Latest session: {sessions[0]}")
    else:
        rec_status, rec_msg = get_recorder_status()
        if rec_status == 'streaming':
            st.success("🟢 **STREAMING**")
            st.caption("Both sensors connected & recording")
        elif rec_status == 'error':
            st.error(f"❌ **Recorder Error**")
            st.caption(rec_msg)
        else:
            st.warning("⏳ **Connecting...**")
            st.caption(rec_msg)

with col2:
    if st.session_state.recording_pid is None:
        if st.button("START SESSION", type="primary", width='stretch'):
            try:
                # Clean up old raw files and log
                for f in [CHEST_FILE, WRIST_FILE, RECORDER_LOG]:
                    if os.path.exists(f):
                        os.remove(f)

                # New session ID
                session_id = time.strftime("%Y%m%d_%H%M%S")
                st.session_state.current_session_id = session_id
                st.session_state.selected_session_id = session_id

                # Reset live history
                st.session_state.live_jump_history = []

                # Wire rt_jumps_file into params so worker can persist RT results
                _, _, rt_file = get_session_files(session_id)
                rt_params['rt_jumps_file'] = rt_file

                # Launch recorder in a hidden window, no pop up
                log_fh = open(RECORDER_LOG, 'w', buffering=1)
                proc = subprocess.Popen(
                    [sys.executable, "-u", RECORDER_SCRIPT],
                    cwd=os.getcwd(),
                    stdout=log_fh,
                    stderr=log_fh,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                st.session_state.recording_pid = proc.pid

                # Start background processing thread
                shared = {'running': True, 'latest_jumps': [], 'last_update': None, 'error': None}
                st.session_state.rt_shared = shared
                thread = threading.Thread(
                    target=realtime_worker,
                    args=(shared, rt_params),
                    daemon=True
                )
                thread.start()
                st.session_state.rt_thread = thread

                st.rerun()

            except PermissionError:
                st.error("⚠️ A background process is still holding the files.")
    else:
        if st.button("STOP & SAVE", type="secondary", width='stretch'):
            if st.session_state.rt_shared:
                st.session_state.rt_shared['running'] = False

            if st.session_state.recording_pid:
                try:
                    os.kill(st.session_state.recording_pid, signal.SIGTERM)
                except OSError:
                    pass
                st.session_state.recording_pid = None

            #run pipeline in post processing too!
            status_box = st.empty()
            status_box.info("⏳ Synchronizing and filtering full session data...")

            session_id = st.session_state.current_session_id or time.strftime("%Y%m%d_%H%M%S")
            st.session_state.current_session_id = session_id
            st.session_state.selected_session_id = session_id
            session_processed_file, session_jumps_file, _ = get_session_files(session_id)

            try:
                if os.path.exists(CHEST_FILE) and os.path.exists(WRIST_FILE):
                    c_df = pd.read_csv(CHEST_FILE, on_bad_lines='skip')
                    w_df = pd.read_csv(WRIST_FILE, on_bad_lines='skip')

                    if len(c_df) > 10 and len(w_df) > 10:
                        synced = align_sensors_zeroed(c_df, w_df)
                        processed = process_imu_data(synced, enable_butterworth=True, cutoff_freq=10.0, sensors='chest_only')
                        processed.to_csv(session_processed_file, index=False)

                        status_box.info("🏃 Analyzing jump metrics...")
                        jumps = detect_jumps(processed, **{
                            'flight_threshold': flight_thresh,
                            'min_takeoff_vel': min_vel,
                            'min_takeoff_accel_g': min_push,
                            'min_landing_accel_g': min_land,
                            'min_height_cm': min_height,
                            'max_height_cm': max_height,
                        })

                        if jumps:
                            jumps = detect_hits(processed, jumps,
                                               min_energy_millions * 1_000_000.0, min_gyro, min_accel)
                        if jumps:
                            jumps = evaluate_hit_timing(jumps, early_thresh, late_thresh)

                        if jumps:
                            pd.DataFrame(jumps).to_csv(session_jumps_file, index=False)
                            status_box.success(f"✅ Session {session_id} saved! {len(jumps)} jumps detected.")
                        else:
                            if os.path.exists(session_jumps_file):
                                os.remove(session_jumps_file)
                            status_box.success(f"✅ Session {session_id} saved. No jumps detected.")

                        time.sleep(2)
                    else:
                        status_box.warning("⚠️ Files too short to process.")
                        time.sleep(2)
                else:
                    status_box.warning("⚠️ No data files found.")
                    time.sleep(2)

            except Exception as e:
                status_box.error(f"❌ Processing Error: {e}")
                time.sleep(5)

            st.rerun()


# LIVE UI DISPLAY!!!
if st.session_state.recording_pid:
    st.markdown("---")

    shared = st.session_state.rt_shared

    latest_jumps = shared.get('latest_jumps', [])

    for jump in latest_jumps:
        t_new = jump.get('takeoff_t', 0)
        already_seen = any(
            abs(t_new - j.get('takeoff_t', 0)) < 0.5
            for j in st.session_state.live_jump_history
        )
        if not already_seen:
            st.session_state.live_jump_history.append(jump)

    history = st.session_state.live_jump_history

    #Most recent jump is big
    if history:
        last = history[-1]
        hit_detected = last.get('hit_detected', False)
        height = last.get('height_cm', 0.0)
        hit_height = last.get('height_at_hit_cm')
        timing = last.get('hit_timing', '')
        impact_g = last.get('hit_peak_g')

        st.subheader("Latest Jump")

        left_col, right_col = st.columns([1, 1])

        with left_col:
            m1, m2 = st.columns(2)
            m1.metric("Jump Height", f"{height:.1f} cm")
            m2.metric("Hit", "🏐 Yes" if hit_detected else "❌ No")
            if hit_detected and hit_height is not None:
                st.metric("Height at Hit", f"{hit_height:.1f} cm  {timing_emoji(timing)} {timing}")
            else:
                st.metric("Height at Hit", "—")
            if hit_detected and impact_g is not None:
                st.caption(f"Impact: {impact_g:.1f} g")

        with right_col:
            parabola = make_parabola_chart(last)
            if parabola is not None:
                st.altair_chart(parabola, width='stretch')

    else:
        rec_status, rec_msg = get_recorder_status()
        if rec_status == 'error':
            st.error(f"❌ **Recorder crashed:** {rec_msg}")
            try:
                with open(RECORDER_LOG, 'r') as f:
                    log_lines = f.readlines()
                st.code(''.join(log_lines[-10:]), language=None)
            except Exception:
                pass
            st.caption("Stop the session and restart.")
        elif rec_status == 'streaming':
            st.success("🟢 **Sensors connected — waiting for first jump...**")
        else:
            st.warning("⏳ Waiting for sensors to connect...")
            st.caption(rec_msg)


# Detect trends based on stat def of trends, not any sports science fatigue def
    if len(history) >= 5:
        st.markdown("---")
        st.markdown("#### 📉 Live Trends & Fatigue")

        t_col1, t_col2 = st.columns([1, 2])

        heights = [j.get('height_cm', 0) for j in history]
        recent_heights = heights[-5:]
        session_avg = sum(heights) / len(heights)

        strictly_decreasing = all(recent_heights[i] < recent_heights[i-1] for i in range(1, 5))
        below_avg_shift = all(h < session_avg for h in recent_heights)

        with t_col1:
            if strictly_decreasing:
                st.warning(f"⚠️ **Fatigue Alert:** 4 consecutive jumps decreasing.")
            elif below_avg_shift:
                st.info(f"📉 **Fatigue Trend:** Last 5 jumps below avg ({session_avg:.1f} cm).")
            else:
                st.success("🟢 Jump height is stable.")

            recent_hits = [j for j in history[-10:] if j.get('hit_detected')]
            if len(recent_hits) >= 4:
                recent_timings = [j.get('hit_timing') for j in recent_hits[-4:]]
                if all(t == 'Early' for t in recent_timings):
                    st.warning("⏫ **Timing Drift:** 4 'Early' hits in a row!")
                elif all(t == 'Late' for t in recent_timings):
                    st.warning("⏬ **Timing Drift:** 4 'Late' hits in a row!")

        with t_col2:
            spark_df = pd.DataFrame({
                'Jump Number': range(1, len(heights) + 1),
                'Height (cm)': heights
            })

            # The line and dots
            line_chart = alt.Chart(spark_df).mark_line(point=True, color="#4c78a8").encode(
                x=alt.X('Jump Number:O', title=None, axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y('Height (cm):Q', scale=alt.Scale(zero=False)),
                tooltip=['Jump Number', 'Height (cm)']
            )
            #session avg
            avg_rule = alt.Chart(pd.DataFrame({'y': [session_avg]})).mark_rule(
                color='red', strokeDash=[3, 3]
            ).encode(y='y:Q')

            spark_combo = (line_chart + avg_rule).properties(height=150)
            st.altair_chart(spark_combo, width='stretch')


    #scrollable
    if len(history) > 1:
        st.markdown("#### Session Log")
        log_rows = []
        for i, j in enumerate(reversed(history), 1):
            hit = j.get('hit_detected', False)
            hh = j.get('height_at_hit_cm')
            tm = j.get('hit_timing', '')
            log_rows.append({
                '#': len(history) - i + 1,
                'Height (cm)': round(j.get('height_cm', 0), 1),
                'Hit': '🏐 Yes' if hit else '❌ No',
                'Hit Height (cm)': round(hh, 1) if hh is not None else None,
                'Timing': f"{timing_emoji(tm)} {tm}" if hit else '—',
            })
        st.dataframe(pd.DataFrame(log_rows), width="stretch", hide_index=True)

    if shared.get('error'):
        st.caption(f"⚠️ Worker: {shared['error']}")

    #UI refresh every 1s, matches rt updates
    time.sleep(1.0)
    st.rerun()


# NOT LIVE View (post processed results)
if st.session_state.recording_pid is None:
    sessions = list_sessions()

    if sessions:
        st.markdown("---")
        st.subheader("📂 Session Browser")

        selected_label = st.selectbox(
            "Select a session to view:",
            options=sessions,
            index=0 if st.session_state.selected_session_id not in sessions
                    else sessions.index(st.session_state.selected_session_id),
            key="session_selector"
        )
        st.session_state.selected_session_id = selected_label

        active_session_id = selected_label
        active_processed_file, active_jumps_file, active_rt_jumps_file = get_session_files(active_session_id)

        if os.path.exists(active_processed_file):

            col_res1, col_res2 = st.columns([3, 1])
            with col_res1:
                st.subheader("🏆 Session Results")
            with col_res2:
                if st.button("🔄 Recalculate", width='stretch'):
                    proc_df = pd.read_csv(active_processed_file)
                    new_jumps = detect_jumps(proc_df, **{
                        'flight_threshold': flight_thresh,
                        'min_takeoff_vel': min_vel,
                        'min_takeoff_accel_g': min_push,
                        'min_landing_accel_g': min_land,
                        'min_height_cm': min_height,
                        'max_height_cm': max_height,
                    })

                    if new_jumps:
                        new_jumps = detect_hits(proc_df, new_jumps,
                                               min_energy_millions * 1_000_000.0, min_gyro, min_accel)
                    if new_jumps:
                        new_jumps = evaluate_hit_timing(new_jumps, early_thresh, late_thresh)

                    try:
                        if new_jumps:
                            pd.DataFrame(new_jumps).to_csv(active_jumps_file, index=False)
                        else:
                            if os.path.exists(active_jumps_file):
                                os.remove(active_jumps_file)
                        st.rerun()
                    except PermissionError:
                        st.error(f"⚠️ Cannot save! Please close `{os.path.basename(active_jumps_file)}` in Excel.")

            # table and graph
            tab_post, tab_rt, tab_compare = st.tabs(
                ["📊 Post-Processing", "⚡ Real-Time (recorded)", "🔍 Compare RT vs Post"]
            )

            with tab_post:
             if os.path.exists(active_jumps_file):
                try:
                    jumps_df = pd.read_csv(active_jumps_file)
                    if not jumps_df.empty:
                        st.success(f"Successfully tracked **{len(jumps_df)}** jumps!")

                        display_df = jumps_df.copy()
                        display_df['Takeoff (s)'] = display_df['takeoff_t'].round(2)
                        display_df['Jump Height (cm)'] = display_df['height_cm'].round(1)
                        display_df['Hit?'] = display_df['hit_detected'].apply(
                            lambda x: '🏐 Yes' if x else '❌ No')

                        if 'height_at_hit_cm' in display_df.columns:
                            display_df['Hit Jump Height (cm)'] = pd.to_numeric(
                                display_df['height_at_hit_cm'], errors='coerce')
                            display_df.loc[display_df['hit_detected'] == False, 'Hit Jump Height (cm)'] = None
                            display_df['Timing'] = display_df['hit_timing'].where(
                                display_df['hit_detected'], other='')

                        if 'hit_peak_g' in display_df.columns:
                            display_df['Hit acc (g)'] = display_df['hit_peak_g'].round(1)
                            display_df.loc[display_df['hit_detected'] == False, 'Hit Impact (g)'] = None

                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            st.metric("Max Jump Height", f"{display_df['Jump Height (cm)'].max():.1f} cm")
                            st.metric("Avg Jump Height", f"{display_df['Jump Height (cm)'].mean():.1f} cm")
                            st.metric("Total Hits", f"{display_df['hit_detected'].sum()}")

                        with col_b:
                            cols_to_show = ['Takeoff (s)', 'Jump Height (cm)', 'Hit?',
                                            'Hit Jump Height (cm)', 'Timing', 'Hit acc (g)']
                            cols_to_show = [c for c in cols_to_show if c in display_df.columns]
                            st.dataframe(display_df[cols_to_show], width='stretch', hide_index=True)

                        #graphs for each jump
                        st.markdown("### 📈 Jump Profiles (Chest & Wrist)")
                        st.caption("Solid red = Max Accel hit. Dashed orange = Max Jerk hit.")

                        proc_df = pd.read_csv(active_processed_file)
                        proc_df['time_s'] = proc_df['timestamp'] / 1000.0
                        proc_df = compute_hit_signals(proc_df)

                        if 'Hit_Energy_Envelope' in proc_df.columns:
                            proc_df['Scaled_Jerk'] = proc_df['Hit_Energy_Envelope'] / 1_000_000.0
                        else:
                            proc_df['Scaled_Jerk'] = 0.0

                        for idx, row in jumps_df.iterrows():
                            hit_status = "🏐 Hit!" if row.get('hit_detected', False) else "❌ No Hit"

                            with st.expander(
                                f"Jump {idx+1} @ {row['takeoff_t']:.1f}s | {row['height_cm']:.1f} cm | {hit_status}"
                            ):
                                t_start = row['takeoff_t'] - 0.5
                                t_end = row['landing_t'] + 0.5
                                mask = (proc_df['time_s'] >= t_start) & (proc_df['time_s'] <= t_end)
                                plot_df = proc_df[mask].set_index('time_s')

                                plot_cols = {}
                                if 'acc_z_global_chest' in plot_df.columns:
                                    plot_cols['acc_z_global_chest'] = 'Chest Vertical (g)'
                                if 'Wrist_Mag_g' in plot_df.columns:
                                    plot_cols['Wrist_Mag_g'] = 'Filtered Wrist Mag (g)'
                                if 'Scaled_Jerk' in plot_df.columns:
                                    plot_cols['Scaled_Jerk'] = 'Jerk Energy (Millions)'

                                if plot_cols:
                                    plot_df = plot_df.rename(columns=plot_cols)
                                    alt_df = plot_df.reset_index()
                                    alt_df_melted = alt_df.melt(
                                        id_vars=['time_s'],
                                        value_vars=list(plot_cols.values()),
                                        var_name='Sensor',
                                        value_name='Value'
                                    )

                                    base_chart = alt.Chart(alt_df_melted).mark_line().encode(
                                        x=alt.X('time_s:Q', title='Time (s)'),
                                        y=alt.Y('Value:Q', title='Acceleration (g) / Energy (M)'),
                                        color=alt.Color('Sensor:N',
                                                        legend=alt.Legend(title="Metrics", orient="bottom"))
                                    )

                                    takeoff_marker = alt.Chart(
                                        pd.DataFrame({'takeoff_t': [row['takeoff_t']]})
                                    ).mark_rule(color='green', strokeWidth=2, strokeDash=[5, 5]).encode(
                                        x='takeoff_t:Q')

                                    landing_marker = alt.Chart(
                                        pd.DataFrame({'landing_t': [row['landing_t']]})
                                    ).mark_rule(color='blue', strokeWidth=2, strokeDash=[5, 5]).encode(
                                        x='landing_t:Q')

                                    final_chart = base_chart + takeoff_marker + landing_marker

                                    if row.get('hit_detected', False):
                                        if pd.notna(row.get('hit_time')):
                                            hit_marker = alt.Chart(
                                                pd.DataFrame({'hit_time': [row['hit_time']]})
                                            ).mark_rule(color='red', strokeWidth=2).encode(x='hit_time:Q')
                                            final_chart += hit_marker

                                        if 'hit_time_jerk' in row and pd.notna(row.get('hit_time_jerk')):
                                            jerk_marker = alt.Chart(
                                                pd.DataFrame({'hit_time_jerk': [row['hit_time_jerk']]})
                                            ).mark_rule(color='orange', strokeWidth=2,
                                                        strokeDash=[3, 3]).encode(x='hit_time_jerk:Q')
                                            final_chart += jerk_marker

                                    st.altair_chart(final_chart, width='stretch')

                except Exception as e:
                    st.error(f"Could not load jump results: {e}")
             else:
                st.info("No jumps detected with current settings. Try lowering the thresholds!")

            #Saved real-time results
            with tab_rt:
                if os.path.exists(active_rt_jumps_file):
                    try:
                        rt_df = pd.read_csv(active_rt_jumps_file)
                        if not rt_df.empty:
                            st.success(f"Real-time pipeline recorded **{len(rt_df)}** jumps during the session.")
                            rt_disp = rt_df.copy()
                            rt_disp['Takeoff (s)'] = rt_disp['takeoff_t'].round(2)
                            rt_disp['Height (cm)'] = rt_disp['height_cm'].round(1)
                            rt_disp['Hit Made'] = rt_disp['hit_detected'].apply(
                                lambda x: '🏐 Yes' if x else '❌ No')
                            if 'height_at_hit_cm' in rt_disp.columns:
                                rt_disp['Hit Height (cm)'] = pd.to_numeric(
                                    rt_disp['height_at_hit_cm'], errors='coerce')
                                rt_disp.loc[rt_disp['hit_detected'] == False, 'Hit Height (cm)'] = None
                                rt_disp['Timing'] = rt_disp['hit_timing'].where(
                                    rt_disp['hit_detected'], other='')
                            if 'hit_peak_g' in rt_disp.columns:
                                rt_disp['Hit Impact (g)'] = rt_disp['hit_peak_g'].round(1)
                                rt_disp.loc[rt_disp['hit_detected'] == False, 'Hit Impact (g)'] = None

                            col_rt1, col_rt2 = st.columns([1, 2])
                            with col_rt1:
                                st.metric("Max Jump Height", f"{rt_disp['Height (cm)'].max():.1f} cm")
                                st.metric("Avg Jump Height", f"{rt_disp['Height (cm)'].mean():.1f} cm")
                                st.metric("Total Hits", f"{rt_disp['hit_detected'].sum()}")
                            with col_rt2:
                                rt_cols = ['Takeoff (s)', 'Height (cm)', 'Hit Made',
                                           'Hit Height (cm)', 'Timing', 'Hit Impact (g)']
                                rt_cols = [c for c in rt_cols if c in rt_disp.columns]
                                st.dataframe(rt_disp[rt_cols], width='stretch', hide_index=True)
                        else:
                            st.info("RT file exists but is empty.")
                    except Exception as e:
                        st.error(f"Could not load RT results: {e}")
                else:
                    st.info("No real-time data saved for this session. RT data is only captured during live recording.")

            #Compare real-time and post (this is good for prototype phase, doesnt need to be in final ui)
            with tab_compare:
                has_post = os.path.exists(active_jumps_file)
                has_rt = os.path.exists(active_rt_jumps_file)

                if not has_post and not has_rt:
                    st.info("No data available for either pipeline yet.")
                else:
                    try:
                        post_df = pd.read_csv(active_jumps_file) if has_post else pd.DataFrame()
                        rt_df   = pd.read_csv(active_rt_jumps_file) if has_rt else pd.DataFrame()

                        #comparison summary
                        st.markdown("#### Summary")
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Jumps detected",
                                   f"Post: {len(post_df)}  |  RT: {len(rt_df)}",
                                   delta=len(rt_df) - len(post_df),
                                   delta_color="off")
                        if not post_df.empty and not rt_df.empty:
                            post_hits = int(post_df['hit_detected'].sum()) if 'hit_detected' in post_df.columns else 0
                            rt_hits   = int(rt_df['hit_detected'].sum())   if 'hit_detected' in rt_df.columns   else 0
                            sc2.metric("Hits detected",
                                       f"Post: {post_hits}  |  RT: {rt_hits}",
                                       delta=rt_hits - post_hits,
                                       delta_color="off")
                            post_avg = post_df['height_cm'].mean() if 'height_cm' in post_df.columns else float('nan')
                            rt_avg   = rt_df['height_cm'].mean()   if 'height_cm' in rt_df.columns   else float('nan')
                            sc3.metric("Avg height (cm)",
                                       f"Post: {post_avg:.1f}  |  RT: {rt_avg:.1f}",
                                       delta=round(rt_avg - post_avg, 1),
                                       delta_color="off")

                        # compare by jump
                        st.markdown("#### Per-jump comparison")
                        st.caption(
                            "Jumps are matched by takeoff time (±0.5 s). "
                            "🟡 = value differs between pipelines."
                        )

                        if post_df.empty:
                            st.warning("No post-processing jumps to compare.")
                        elif rt_df.empty:
                            st.warning("No real-time jumps to compare.")
                        else:
                            compare_rows = []
                            used_rt = set()

                            for pi, pr in post_df.iterrows():
                                pt = pr.get('takeoff_t', np.nan)
                                # Find closest RT jump within 0.5s
                                match_i = None
                                best_dt = 0.5
                                for ri, rr in rt_df.iterrows():
                                    if ri in used_rt:
                                        continue
                                    dt = abs(rr.get('takeoff_t', np.nan) - pt)
                                    if dt < best_dt:
                                        best_dt = dt
                                        match_i = ri

                                if match_i is not None:
                                    used_rt.add(match_i)
                                    rr = rt_df.loc[match_i]
                                    p_h = pr.get('height_cm', np.nan)
                                    r_h = rr.get('height_cm', np.nan)
                                    p_hit = pr.get('hit_detected', False)
                                    r_hit = rr.get('hit_detected', False)
                                    p_timing = pr.get('hit_timing', '—')
                                    r_timing = rr.get('hit_timing', '—')
                                    p_hh = pr.get('height_at_hit_cm', np.nan)
                                    r_hh = rr.get('height_at_hit_cm', np.nan)
                                    p_ft = pr.get('flight_time_s', np.nan)
                                    r_ft = rr.get('flight_time_s', np.nan)
                                    p_land = pr.get('landing_t', np.nan)
                                    r_land = rr.get('landing_t', np.nan)
                                    ft_diff = round(r_ft - p_ft, 3) if pd.notna(p_ft) and pd.notna(r_ft) else None


                                    h_flag  = '🟡' if abs(p_h - r_h) > 2 else ''
                                    hit_flag = '🟡' if p_hit != r_hit else ''
                                    tim_flag = '🟡' if p_timing != r_timing else ''
                                    hh_flag  = ''
                                    if pd.notna(p_hh) and pd.notna(r_hh):
                                        hh_flag = '🟡' if abs(p_hh - r_hh) > 2 else ''

                                    compare_rows.append({
                                        'Jump': str(pi + 1),
                                        'Takeoff (s)': round(pt, 2),
                                        'Height Post (cm)': round(p_h, 1) if pd.notna(p_h) else None,
                                        'Height RT (cm)': round(r_h, 1) if pd.notna(r_h) else None,
                                        'Height Diff': h_flag,
                                        'Hit Post': '🏐' if p_hit else '❌',
                                        'Hit RT': '🏐' if r_hit else '❌',
                                        'Hit Diff': hit_flag,
                                        'Hit Height Post (cm)': round(p_hh, 1) if pd.notna(p_hh) else None,
                                        'Hit Height RT (cm)': round(r_hh, 1) if pd.notna(r_hh) else None,
                                        'Timing Post': p_timing if p_hit else '',
                                        'Timing RT': r_timing if r_hit else '',
                                        'Timing Diff': tim_flag,
                                        'Landing Post (s)': round(p_land, 2) if pd.notna(p_land) else None,
                                        'Landing RT (s)': round(r_land, 2) if pd.notna(r_land) else None,
                                        'FT Post (s)': round(p_ft, 3) if pd.notna(p_ft) else None,
                                        'FT RT (s)': round(r_ft, 3) if pd.notna(r_ft) else None,
                                        'FT Diff (s)': ft_diff,
                                    })
                                else:
                                    compare_rows.append({
                                        'Jump': str(pi + 1),
                                        'Takeoff (s)': round(pt, 2),
                                        'Height Post (cm)': round(pr.get('height_cm', float('nan')), 1),
                                        'Height RT (cm)': None,
                                        'Height Diff': '🟡',
                                        'Hit Post': '🏐' if pr.get('hit_detected') else '❌',
                                        'Hit RT': '—',
                                        'Hit Diff': '',
                                        'Hit Height Post (cm)': None, 'Hit Height RT (cm)': None,
                                        'Timing Post': '', 'Timing RT': '',
                                        'Timing Diff': '',
                                        'Landing Post (s)': None,
                                        'Landing RT (s)': None,
                                        'FT Post (s)': round(pr.get('flight_time_s', float('nan')), 3),
                                        'FT RT (s)': None,
                                        'FT Diff (s)': None,
                                    })

                            # RT jumps with no post match
                            for ri, rr in rt_df.iterrows():
                                if ri not in used_rt:
                                    compare_rows.append({
                                        'Jump': 'RT only',
                                        'Takeoff (s)': round(rr.get('takeoff_t', float('nan')), 2),
                                        'Height Post (cm)': None,
                                        'Height RT (cm)': round(rr.get('height_cm', float('nan')), 1),
                                        'Height Diff': '🟡',
                                        'Hit Post': '—',
                                        'Hit RT': '🏐' if rr.get('hit_detected') else '❌',
                                        'Hit Diff': '',
                                        'Hit Height Post (cm)': None, 'Hit Height RT (cm)': None,
                                        'Timing Post': '', 'Timing RT': '',
                                        'Timing Diff': '',
                                        'Landing Post (s)': None,
                                        'Landing RT (s)': round(rr.get('landing_t', float('nan')), 2),
                                        'FT Post (s)': None,
                                        'FT RT (s)': round(rr.get('flight_time_s', float('nan')), 3),
                                        'FT Diff (s)': None,
                                    })

                            if compare_rows:
                                st.dataframe(pd.DataFrame(compare_rows),
                                             width='stretch', hide_index=True)

                    except Exception as e:
                        st.error(f"Comparison error: {e}")
                    



  # Long term trends - again stat trends to indicate 'fatigue'
    st.markdown("---")
    st.header("📈 Long-Term Trends")
    st.caption("Based on previous stored real-time session data")

    all_rt_jumps = []
    #oldest to newest
    for s_id in reversed(sessions):
        _, _, rt_file = get_session_files(s_id)
        if os.path.exists(rt_file):
            try:
                df = pd.read_csv(rt_file)
                if not df.empty and 'height_cm' in df.columns:
                    # Parse session ID to a readable datetime
                    s_date = pd.to_datetime(s_id, format='%Y%m%d_%H%M%S')
                    df['Session Date'] = s_date
                    df['Session ID'] = s_id
                    all_rt_jumps.append(df)
            except Exception:
                pass

    if all_rt_jumps:
        hist_df = pd.concat(all_rt_jumps, ignore_index=True)

        # only use jumps with hits
        if 'hit_detected' in hist_df.columns:
            hist_df = hist_df[hist_df['hit_detected'] == True].copy()

        # get rid of files with less tan 10 jumps bc that was probably a quick test session
        if not hist_df.empty:
            session_hit_counts = hist_df.groupby('Session ID').size()
            valid_session_ids = session_hit_counts[session_hit_counts >= 10].index
            hist_df = hist_df[hist_df['Session ID'].isin(valid_session_ids)]

        if hist_df.empty:
            st.info("No full attack sessions (10+ hits) found in historical data yet.")
        else:
            # Remove outliers using IQR
            Q1 = hist_df['height_cm'].quantile(0.25)
            Q3 = hist_df['height_cm'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            hist_df = hist_df[(hist_df['height_cm'] >= lower_bound) & (hist_df['height_cm'] <= upper_bound)]

            if hist_df.empty:
                st.info("No valid attack jumps remaining after outlier removal.")
            else:
                #summary of the avg sessions
                session_stats = hist_df.groupby(['Session Date', 'Session ID']).agg(
                    Avg_Height=('height_cm', 'mean'),
                    Max_Height=('height_cm', 'max'),
                    Attack_Count=('height_cm', 'count')
                ).reset_index()

                session_stats = session_stats.sort_values('Session Date').reset_index(drop=True)
                session_stats['Session Number'] = session_stats.index + 1

                #date label
                session_stats['Session Label'] = "S" + session_stats['Session Number'].astype(str) + " (" + session_stats['Session Date'].dt.strftime('%b %d') + ")"

                # overall
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Attacks Tracked", len(hist_df))
                c2.metric("All-Time Max Attack", f"{hist_df['height_cm'].max():.1f} cm")
                c3.metric("All-Time Avg Attack", f"{hist_df['height_cm'].mean():.1f} cm")

                #trend rules
                if len(session_stats) >= 3:
                    last_3_avgs = session_stats['Avg_Height'].tail(3).tolist()

                    if last_3_avgs[2] < last_3_avgs[1] < last_3_avgs[0]:
                        c4.metric("Performance Trend", "⚠️ Fatigue", help="Average attack height dropped consecutively over your last 3 sessions.")
                    elif last_3_avgs[2] > last_3_avgs[1] > last_3_avgs[0]:
                        c4.metric("Performance Trend", "🔥 Improving", help="Average attack height increased consecutively over your last 3 sessions.")
                    else:
                        c4.metric("Performance Trend", "✅ Stable", help="Attack height is fluctuating normally.")
                else:
                    c4.metric("Performance Trend", "Not enough data", help="Need at least 3 full sessions to calculate a trend.")

                st.markdown("#### Session-by-Session Attack Height")

                chart_df = session_stats.melt(
                    id_vars=['Session Label', 'Session Number', 'Session Date'],
                    value_vars=['Avg_Height', 'Max_Height'],
                    var_name='Metric',
                    value_name='Height (cm)'
                )

                chart_df['Metric'] = chart_df['Metric'].replace({
                    'Avg_Height': 'Average Height',
                    'Max_Height': 'Max Height'
                })

                trend_chart = alt.Chart(chart_df).mark_line(
                    point=True,
                    interpolate='monotone'
                ).encode(
                    x=alt.X(
                        'Session Label:O',
                        sort=alt.EncodingSortField(field='Session Number', order='ascending'),
                        title='Session',
                        axis=alt.Axis(labelAngle=-45)
                    ),
                    y=alt.Y('Height (cm):Q', scale=alt.Scale(zero=False)),
                    color=alt.Color('Metric:N', legend=alt.Legend(title=None, orient="bottom")),
                    tooltip=[
                        alt.Tooltip('Session Date:T', title='Exact Time', format='%Y-%m-%d %H:%M'),
                        'Metric:N',
                        alt.Tooltip('Height (cm):Q', format='.1f')
                    ]
                ).properties(height=300)

                st.altair_chart(trend_chart, width='stretch')

    else:
        st.info("No real-time jump data found. Sessions must contain at least 10 hits to be included here.")