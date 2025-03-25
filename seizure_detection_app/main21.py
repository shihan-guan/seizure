import streamlit as st
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import torch
import torch.nn as nn
import os
import tempfile
import time

from io import BytesIO
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.patches import Rectangle

# Import your models
from model.eegcnn import EEGCNN
from model.transformercnn import EEGTransformerCNN

# ----- Custom CSS -----
CUSTOM_CSS = """
<style>
/* Use a clean sans-serif font (e.g. "SF Pro Display" if installed, fallback to system sans) */
html, body, [class*="css"]  {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif;
}

/* Make headings thinner, bigger, and more Apple-like */
h1, h2, h3, h4 {
  font-weight: 500;
  color: #222;
}

/* Card-like backgrounds for the main container */
section.main > div {
  background-color: #fafafa;
  padding: 2rem;
  border-radius: 8px;
}

/* Subtle transitions for interactive elements */
button, input, select, textarea {
  transition: all 0.2s ease;
}

/* A subtle hover effect for buttons */
button:hover {
  background-color: #f0f0f0 !important;
}

/* Make our pipeline step tabs pop a bit */
div[role="tablist"] > div {
  background-color: #f4f4f4;
  border-radius: 6px;
  padding: 0.5rem;
}
div[role="tab"] {
  border-radius: 4px !important;
  transition: background-color 0.2s;
}
div[role="tab"]:hover {
  background-color: #e8e8e8 !important;
}

/* Style the "success" / "error" messages more softly */
.element-container > .stAlert {
  border-radius: 6px;
}
.reportview-container .stAlert > div {
  padding: 0.75rem 1rem !important;
}

/* Make snippet plots and charts a bit bigger */
.css-1aumxhk {
  max-width: 1200px !important;
}
</style>
"""

# ----- Pipeline Functions -----
BIPOLAR_PAIRS = [
    ('Fp2', 'T4'),
    ('T4', 'O2'),
    ('Fp2', 'C4'),
    ('C4', 'O2'),
    ('T4', 'C4'),
    ('C4', 'Cz'),
    ('Cz', 'C3'),
    ('C3', 'T3'),
    ('Fp1', 'T3'),
    ('T3', 'O1'),
    ('Fp1', 'C3'),
    ('C3', 'O1'),
]

def parse_seizure_annotations(raw):
    intervals = []
    for ann in raw.annotations:
        desc = ann['description'].lower()
        dur = ann['duration']
        if ("crise" in desc or "seizure" in desc) and dur > 0:
            intervals.append((ann['onset'], dur))
    return intervals

def apply_bipolar_montage(raw, bipolar_pairs):
    new_data = []
    new_ch_names = []
    for (ch1, ch2) in bipolar_pairs:
        if ch1 in raw.ch_names and ch2 in raw.ch_names:
            idx1 = raw.ch_names.index(ch1)
            idx2 = raw.ch_names.index(ch2)
            bp_data = raw._data[idx1] - raw._data[idx2]
            new_data.append(bp_data)
            new_ch_names.append(f"{ch1}-{ch2}")
    if not new_data:
        return None
    new_data = np.array(new_data)
    info = mne.create_info(new_ch_names, raw.info['sfreq'], ch_types='eeg')
    return mne.io.RawArray(new_data, info, verbose=False)

def bandpass_filter(raw, l_freq=0.5, h_freq=70.0):
    return raw.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)

def notch_filter(raw, freqs=None):
    if freqs and len(freqs) > 0:
        return raw.copy().notch_filter(freqs=freqs, verbose=False)
    else:
        return raw.copy()

def normalize_global_std(raw):
    data = raw.get_data()
    global_std = np.std(data)
    if global_std == 0:
        global_std = 1e-6
    new_data = data / global_std
    new_raw = raw.copy()
    new_raw._data = new_data
    return new_raw

def normalize_channel_wise(raw):
    data = raw.get_data()
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    std[std == 0] = 1e-6
    new_data = (data - mean) / std
    new_raw = raw.copy()
    new_raw._data = new_data
    return new_raw

def normalize_zscore(raw):
    data = raw.get_data()
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    std[std == 0] = 1e-6
    new_data = (data - mean) / std
    new_raw = raw.copy()
    new_raw._data = new_data
    return new_raw

def merge_intervals(intervals, max_gap=2.0):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    cs, cd = intervals[0]
    ce = cs + cd
    for i in range(1, len(intervals)):
        s, d = intervals[i]
        e = s + d
        if s - ce <= max_gap:
            new_end = max(ce, e)
            cd = new_end - cs
            ce = new_end
        else:
            merged.append((cs, cd))
            cs, cd = s, d
            ce = s + d
    merged.append((cs, cd))
    return merged

def complement_intervals(total_dur, intervals):
    if not intervals:
        return [(0.0, total_dur)]
    intervals = sorted(intervals, key=lambda x: x[0])
    result = []
    prev_end = 0.0
    for (start, dur) in intervals:
        end = start + dur
        if start > prev_end:
            result.append((prev_end, start - prev_end))
        prev_end = max(prev_end, end)
    if prev_end < total_dur:
        result.append((prev_end, total_dur - prev_end))
    return result

def compute_overlap_duration(gt_intervals, pred_intervals):
    total_overlap = 0.0
    for gt_onset, gt_dur in gt_intervals:
        gt_end = gt_onset + gt_dur
        for pred_onset, pred_dur in pred_intervals:
            pred_end = pred_onset + pred_dur
            overlap = max(0, min(gt_end, pred_end) - max(gt_onset, pred_onset))
            total_overlap += overlap
    return total_overlap

# ----- Plotting Functions -----
def plot_freq_domain_comparison(raw_before, raw_after,
                                title_before="Before",
                                title_after="After",
                                fmin=0.1, fmax=100.0):
    psd_bef, freqs_bef = raw_before.compute_psd(fmin=fmin, fmax=fmax).get_data(return_freqs=True)
    psd_aft, freqs_aft = raw_after.compute_psd(fmin=fmin, fmax=fmax).get_data(return_freqs=True)
    avg_psd_bef = psd_bef.mean(axis=0)
    avg_psd_aft = psd_aft.mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(7,2.5))
    axes[0].semilogy(freqs_bef, avg_psd_bef)
    axes[0].set_title(title_before, fontsize=10)
    axes[0].set_xlabel("Frequency (Hz)", fontsize=9)
    axes[0].set_ylabel("PSD (log)", fontsize=9)
    axes[1].semilogy(freqs_aft, avg_psd_aft)
    axes[1].set_title(title_after, fontsize=10)
    axes[1].set_xlabel("Frequency (Hz)", fontsize=9)
    plt.tight_layout()
    return fig

def show_basic_info_panel(step_number, step_name, raw_obj, intervals):
    with st.sidebar.expander(f"Step {step_number}: {step_name} Info", expanded=False):
        if raw_obj is None:
            st.info("No data loaded yet for this step.")
            return
        st.write(f"**Num Channels:** {raw_obj.info['nchan']}")
        st.write(f"**Channel Names:** {raw_obj.ch_names}")
        st.write(f"**Sampling Freq:** {raw_obj.info['sfreq']} Hz")
        st.write(f"**Data Shape:** {raw_obj.get_data().shape}")
        st.write(f"**Duration:** {raw_obj.times[-1]:.2f} s")
        if intervals:
            st.write("**Seizure Intervals (from EDF Annotations):**")
            for i, (onset, dur) in enumerate(intervals):
                st.write(f"- #{i}: start={onset:.2f}s, dur={dur:.2f}s")
        else:
            st.write("_No seizure intervals_")

def plot_entire_eeg_multichannel_two_color(raw_obj,
                                           channels_list,
                                           red_intervals=None,
                                           green_intervals=None,
                                           offset_factor=300.0,
                                           decimate=True,
                                           dynamic_offset=False,
                                           red_alpha=0.2,
                                           green_alpha=0.2,
                                           scale='uV',
                                           force_predict_range=False,
                                           return_fig=False,
                                           show_rangeslider=True):
    times = raw_obj.times
    data = raw_obj.get_data()
    fig = go.Figure()

    # By default we keep original decimation logic:
    if decimate:
        max_points = 20000
        dec_factor = max(1, len(times)//max_points)
    else:
        dec_factor = 1

    if scale=='uV':
        data_scaled = data * 1e6
        yaxis_label = "Amplitude (µV + offset)"
    else:
        data_scaled = data
        yaxis_label = "Amplitude (Normalized + offset)"
    if dynamic_offset and channels_list:
        selected = data_scaled[channels_list, :]
        base_offset = np.max(np.abs(selected)) * 1.2
        if base_offset < 1e-9:
            base_offset = 1.0
    else:
        base_offset = offset_factor

    for i, ch_idx in enumerate(channels_list):
        ch_name = raw_obj.ch_names[ch_idx]
        ch_data = data_scaled[ch_idx][::dec_factor]
        t_plot = times[::dec_factor]
        offset = base_offset * i
        ch_data_offset = ch_data + offset
        fig.add_trace(go.Scatter(x=t_plot, y=ch_data_offset,
                                 mode='lines',
                                 line=dict(shape='linear'),
                                 connectgaps=True,
                                 name=ch_name))

    if channels_list:
        v_min = -base_offset * 0.5
        v_max = base_offset * (len(channels_list)-0.5)
    else:
        v_min, v_max = 0,1

    if red_intervals:
        for (onset, dur) in red_intervals:
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=onset, x1=onset+dur,
                          y0=v_min, y1=v_max,
                          fillcolor=f"rgba(255,0,0,{red_alpha})",
                          layer="below", line_width=0)
    if green_intervals:
        for (onset, dur) in green_intervals:
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=onset, x1=onset+dur,
                          y0=v_min, y1=v_max,
                          fillcolor=f"rgba(0,255,0,{green_alpha})",
                          layer="below", line_width=0)

    if force_predict_range:
        v_min, v_max = -2.0, 2.0

    fig.update_layout(title="EEG Multi-Channel Overview",
                      xaxis=dict(title="Time (s)", rangeslider=dict(visible=show_rangeslider)),
                      yaxis=dict(title=yaxis_label, range=[v_min, v_max]))
    return fig if return_fig else fig

def plot_snippet_matplotlib(raw_obj, start_time, duration,
                            intervals_red=None, intervals_green=None,
                            red_alpha=0.2, green_alpha=0.2, title="EEG Snippet"):
    snippet = raw_obj.copy().crop(tmin=start_time, tmax=start_time+duration)
    data, times = snippet.get_data(return_times=True)
    if data.size == 0:
        return None
    data_uV = data * 1e6
    fig, ax = plt.subplots(figsize=(10,4))
    offset = np.max(np.abs(data_uV)) * 1.2 if data_uV.size else 1.0
    for i in range(data_uV.shape[0]):
        ax.plot(times, data_uV[i] + i*offset, label=snippet.ch_names[i])
    if intervals_red:
        for (onset, dur) in intervals_red:
            x0, x1 = onset, onset + dur
            if x1 > start_time and x0 < start_time+duration:
                ax.axvspan(max(x0, start_time), min(x1, start_time+duration),
                           color='red', alpha=red_alpha)
    if intervals_green:
        for (onset, dur) in intervals_green:
            x0, x1 = onset, onset + dur
            if x1 > start_time and x0 < start_time+duration:
                ax.axvspan(max(x0, start_time), min(x1, start_time+duration),
                           color='green', alpha=green_alpha)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV + offset)")
    ax.legend(fontsize='xx-small', loc='upper right')
    plt.tight_layout()
    return fig

# ----- Predict & GIF Functions -----
def predict_all_windows(raw_obj, model, window_sec=3.0, overlap_sec=1.0):
    data = raw_obj.get_data()
    sfreq = raw_obj.info['sfreq']
    total_samples = data.shape[1]
    w_samps = int(window_sec * sfreq)
    step_samps = int((window_sec - overlap_sec) * sfreq)
    start_samp = 0
    intervals = []
    model.eval()
    while (start_samp + w_samps) <= total_samples:
        seg_data = data[:, start_samp:(start_samp+w_samps)]
        t0 = start_samp / sfreq
        t1 = (start_samp + w_samps) / sfreq
        seg_tensor = torch.tensor(seg_data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model(seg_tensor)
            label = torch.argmax(out, dim=1).item()  # 0=nonseiz, 1=seizure
        if label == 1:
            intervals.append((t0, t1-t0, "seizure"))
        else:
            intervals.append((t0, t1-t0, "nonseiz"))
        start_samp += step_samps
    return intervals

def create_sliding_gif(raw_obj, intervals, picks=None, scale='norm', fps=10, dynamic_offset=True):
    """
    This function was originally for 6.2. We remove usage of it in the final code,
    but keep definition as per 'do not change any other part' instruction.
    """
    import numpy as np
    if picks is None:
        picks = list(range(len(raw_obj.ch_names)))
    data = raw_obj.get_data()
    times = raw_obj.times

    max_points = 2000
    dec_factor = max(1, len(times) // max_points)
    t_plot = times[::dec_factor]

    if scale == 'norm':
        data_scaled = data[:, ::dec_factor]
        y_label = "Amplitude (Normalized)"
    else:
        data_scaled = data[:, ::dec_factor] * 1e6
        y_label = "Amplitude (µV)"

    if dynamic_offset:
        selected = data_scaled[picks, :]
        base_offset = np.max(np.abs(selected)) * 1.2
        if base_offset < 1e-9:
            base_offset = 1.0
    else:
        base_offset = 300.0

    traces = []
    for i, ch_idx in enumerate(picks):
        offset = base_offset * i
        traces.append(go.Scatter(
            x=t_plot,
            y=data_scaled[ch_idx] + offset,
            mode='lines',
            name=raw_obj.ch_names[ch_idx],
            line=dict(shape='linear')
        ))

    if picks:
        v_min = -base_offset * 0.5
        v_max = base_offset * (len(picks) - 0.5)
    else:
        v_min, v_max = 0, 1

    layout = go.Layout(
        title="EEG Multi-Channel Overview with Sliding Window",
        xaxis=dict(title="Time (s)", rangeslider=dict(visible=False)),
        yaxis=dict(title=y_label, range=[v_min, v_max]),
        showlegend=True,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {
                    "frame": {"duration": 1000/fps, "redraw": False},
                    "fromcurrent": True,
                    "transition": {"duration": 0}
                }]
            }]
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {"prefix": "Frame: "},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "visible": False,
            "steps": [{"args": [[str(k)], {"frame": {"duration": 1000/fps, "redraw": False}, "mode": "immediate"}],
                       "label": str(k+1), "method": "animate"} for k in range(len(intervals))]
        }]
    )

    frames = []
    for i in range(len(intervals)):
        frame_shapes = []
        for j in range(i + 1):
            st_time, dur, lbl = intervals[j]
            color = 'rgba(255,0,0,0.2)' if lbl == 'seizure' else 'rgba(0,255,0,0.2)'
            frame_shapes.append({
                "type": "rect",
                "xref": "x",
                "yref": "y",
                "x0": st_time,
                "x1": st_time + dur,
                "y0": v_min,
                "y1": v_max,
                "fillcolor": color,
                "line": {"width": 0}
            })
        frames.append(go.Frame(layout={"shapes": frame_shapes}, name=str(i)))

    fig = go.Figure(data=traces, layout=layout, frames=frames)

    fig.update_layout(
        transition={"duration": 0},
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig

# ----- Sidebar Prediction Info -----
def show_predict_info_sidebar():
    with st.sidebar.expander("Step 6: Predict Info", expanded=False):
        if "window_preds" in st.session_state and st.session_state.window_preds:
            raw_seiz = [(s, d) for (s, d, l) in st.session_state.window_preds if l=='seizure']
            st.info(f"Raw seizure segments: {len(raw_seiz)}, total raw duration: {sum(d for (_, d) in raw_seiz):.2f}s (overlap included)")
        else:
            st.info("No raw window predictions yet.")
        if "pred_seizure_intervals" in st.session_state and st.session_state.pred_seizure_intervals:
            merges = st.session_state.pred_seizure_intervals
            st.write(f"**Merged Seizure Intervals** (count={len(merges)})")
            for i, (onset, dur) in enumerate(merges):
                st.write(f"#{i}: onset={onset:.2f}s, dur={dur:.2f}s")
            st.info(f"Merged total seizure duration: {sum(d for (_, d) in merges):.2f}s (no overlap)")
        else:
            st.info("No merged intervals yet.")

# ----- MAIN APP -----
def main():
    st.set_page_config(layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Neonatal seizure detection by LTSI")

    # Initialize session state
    if "raw_original" not in st.session_state:
        st.session_state.raw_original = None
    if "seizure_intervals" not in st.session_state:
        st.session_state.seizure_intervals = []
    if "raw_montaged" not in st.session_state:
        st.session_state.raw_montaged = None
    if "raw_bandpassed" not in st.session_state:
        st.session_state.raw_bandpassed = None
    if "raw_notched" not in st.session_state:
        st.session_state.raw_notched = None
    if "raw_normalized" not in st.session_state:
        st.session_state.raw_normalized = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "window_preds" not in st.session_state:
        st.session_state.window_preds = []
    if "pred_seizure_intervals" not in st.session_state:
        st.session_state.pred_seizure_intervals = []
    if "pred_nonseizure_intervals" not in st.session_state:
        st.session_state.pred_nonseizure_intervals = []

    # Sidebar: pick model
    st.sidebar.header("Pick Model")
    model_opt = st.sidebar.selectbox("Model Type", ["model1", "EEGTransformerCNN"], key="model_opt_select")
    if model_opt == "model1":
        f_eegcnn = st.sidebar.file_uploader("model1.pth", type=["pth", "pt"], key="eegcnn_file")
        if f_eegcnn:
            st.session_state.model = EEGCNN(num_classes=2)
            try:
                sd = torch.load(f_eegcnn, map_location="cpu")
                st.session_state.model.load_state_dict(sd)
                st.sidebar.success("Model loaded!")
            except Exception as e:
                st.sidebar.error(f"Load error: {e}")
        else:
            # Attempt to load default model if no file uploaded
            default_model_path = os.path.join(os.getcwd(), "/mount/src/seizure/seizure_detection_app/best_model_EEGCNN_fold1.pth")
            if os.path.exists(default_model_path):
                st.session_state.model = EEGCNN(num_classes=2)
                try:
                    sd = torch.load(default_model_path, map_location="cpu")
                    st.session_state.model.load_state_dict(sd)
                    st.sidebar.success("Default model loaded!")
                except Exception as e:
                    st.sidebar.error(f"Default model load error: {e}")
            else:
                st.sidebar.info("No default model file found.")
    else:
        st.sidebar.write("Using EEGTransformerCNN.")
        f_transformer = st.sidebar.file_uploader("Transformer .pth", type=["pth", "pt"], key="transformer_file")
        if f_transformer:
            st.session_state.model = EEGTransformerCNN(
                num_channels=12,
                seq_length=768,
                patch_size=8,
                embed_dim=512,
                num_heads=2,
                num_layers=6,
                num_classes=2,
                dropout=0.2
            )
            try:
                sd = torch.load(f_transformer, map_location="cpu")
                st.session_state.model.load_state_dict(sd)
                st.sidebar.success("EEGTransformerCNN loaded!")
            except Exception as e:
                st.sidebar.error(f"Load error: {e}")

    # Tabs
    tabs = st.tabs([
        "1. Load & Overview",
        "2. Montage",
        "3. Bandpass",
        "4. Notch",
        "5. Normalize",
        "6. Predict"
    ])

    # STEP 1: Load & Overview
    with tabs[0]:
        st.subheader("1. Load EDF & Overview")
        edf_up = st.file_uploader("Upload EDF", type=["edf"])
        if edf_up:
            raw_bytes = edf_up.read()
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                tmp.write(raw_bytes)
                tmp.flush()
                path_tmp = tmp.name
            try:
                raw = mne.io.read_raw_edf(path_tmp, preload=True, verbose=False)
                st.session_state.raw_original = raw
                intervals = parse_seizure_annotations(raw)
                st.session_state.seizure_intervals = intervals
                st.success(f"Loaded {edf_up.name}, channels={len(raw.ch_names)}.")
            except Exception as e:
                st.error(f"Cannot load EDF: {e}")
            finally:
                os.remove(path_tmp)
        show_basic_info_panel(1, "Original EDF", st.session_state.raw_original, st.session_state.seizure_intervals)
        if st.session_state.raw_original is not None:
            ro = st.session_state.raw_original
            with st.expander("Multi-Channel Overview (Original)", expanded=True):
                dec_off = st.checkbox("Disable decimation?", value=False)
                all_chs = list(range(len(ro.ch_names)))
                picks = st.multiselect("Channels (Original)", all_chs, default=[0],
                                       format_func=lambda i: ro.ch_names[i])
                off_f = st.number_input("Offset factor (Orig)", 50.0, 2000.0, 300.0, step=50.0)
                if picks:
                    fig = plot_entire_eeg_multichannel_two_color(
                        ro, picks,
                        red_intervals=st.session_state.seizure_intervals,
                        green_intervals=None,
                        offset_factor=off_f, decimate=not dec_off,
                        dynamic_offset=False, scale='uV',
                        show_rangeslider=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with st.expander("Snippet View (Matplotlib)", expanded=True):
                dur_ = st.slider("Snippet Dur (Orig)", 1.0, min(30.0, ro.times[-1]), 5.0)
                stt_ = st.slider("Snippet Start (Orig)", 0.0, float(ro.times[-1]-dur_), 0.0)
                if st.button("Plot Original Snippet"):
                    sfig = plot_snippet_matplotlib(
                        ro, stt_, dur_,
                        intervals_red=st.session_state.seizure_intervals,
                        intervals_green=None,
                        title="Original Snippet"
                    )
                    if sfig:
                        st.pyplot(sfig)

    # STEP 2: Montage
    with tabs[1]:
        st.subheader("2. Montage")
        if st.session_state.raw_original is None:
            st.warning("Load in step 1 first.")
        else:
            if st.button("Apply Bipolar Montage"):
                mraw = apply_bipolar_montage(st.session_state.raw_original, BIPOLAR_PAIRS)
                if mraw is None:
                    st.error("Montage failed: no valid pairs.")
                else:
                    st.session_state.raw_montaged = mraw
                    st.success(f"Bipolar montage success, channels={len(mraw.ch_names)}")
            show_basic_info_panel(2, "Montaged", st.session_state.raw_montaged, st.session_state.seizure_intervals)
            if st.session_state.raw_montaged is not None:
                ro = st.session_state.raw_montaged
                with st.expander("Montaged Multi-Channel Overview", expanded=True):
                    all_chs = list(range(len(ro.ch_names)))
                    picks = st.multiselect("Channels (Montaged)", all_chs, default=all_chs,
                                           format_func=lambda i: ro.ch_names[i])
                    off_f = st.number_input("Offset factor (Montaged)", 50.0, 2000.0, 300.0, step=50.0)
                    if picks:
                        figm = plot_entire_eeg_multichannel_two_color(
                            ro, picks,
                            red_intervals=st.session_state.seizure_intervals,
                            green_intervals=None,
                            offset_factor=off_f, decimate=True,
                            dynamic_offset=False, scale='uV',
                            show_rangeslider=False
                        )
                        st.plotly_chart(figm, use_container_width=True)
                with st.expander("Montaged Snippet", expanded=True):
                    dur__ = st.slider("Snippet Dur (Montaged)", 1.0, min(30.0, ro.times[-1]), 5.0)
                    stt__ = st.slider("Snippet Start (Montaged)", 0.0, float(ro.times[-1]-dur__), 0.0)
                    if st.button("Plot Montaged Snippet"):
                        sfig = plot_snippet_matplotlib(
                            ro, stt__, dur__,
                            intervals_red=st.session_state.seizure_intervals,
                            intervals_green=None,
                            title="Montaged Snippet"
                        )
                        if sfig:
                            st.pyplot(sfig)

    # STEP 3: Bandpass
    with tabs[2]:
        st.subheader("3. Bandpass Filter")
        if st.session_state.raw_montaged is None:
            st.warning("Finish step 2.")
        else:
            lf = st.number_input("Low Freq (Hz)", 0.1, 10.0, 0.5, step=0.1)
            hf = st.number_input("High Freq (Hz)", 10.0, 200.0, 70.0, step=1.0)
            if st.button("Apply Bandpass"):
                bp_r = bandpass_filter(st.session_state.raw_montaged, lf, hf)
                st.session_state.raw_bandpassed = bp_r
                st.success("Bandpass done.")
            show_basic_info_panel(3, "Bandpassed", st.session_state.raw_bandpassed, st.session_state.seizure_intervals)
            if st.session_state.raw_bandpassed is not None:
                ro = st.session_state.raw_bandpassed
                with st.expander("Bandpassed Multi-Channel Overview", expanded=True):
                    all_chs = list(range(len(ro.ch_names)))
                    picks = st.multiselect("Channels (BP)", all_chs, default=[0],
                                           format_func=lambda i: ro.ch_names[i])
                    off_f = st.number_input("Offset factor (BP)", 50.0, 2000.0, 300.0, step=50.0)
                    if picks:
                        figbp = plot_entire_eeg_multichannel_two_color(
                            ro, picks,
                            red_intervals=st.session_state.seizure_intervals,
                            green_intervals=None,
                            offset_factor=off_f, decimate=True,
                            dynamic_offset=False, scale='uV',
                            show_rangeslider=False
                        )
                        st.plotly_chart(figbp, use_container_width=False, height=300)
                with st.expander("Compare PSD: Before & After Bandpass", expanded=True):
                    ff = plot_freq_domain_comparison(
                        st.session_state.raw_montaged, st.session_state.raw_bandpassed,
                        "Before BP", "After BP"
                    )
                    st.pyplot(ff)

    # STEP 4: Notch
    with tabs[3]:
        st.subheader("4. Notch Filter (Default=50)")
        if st.session_state.raw_bandpassed is None:
            st.warning("Finish step 3.")
        else:
            notch_str = st.text_input("Notch freq (comma e.g. 50,60)", "50")
            if st.button("Apply Notch"):
                if notch_str.strip():
                    try:
                        freqs = [float(x.strip()) for x in notch_str.split(",")]
                    except:
                        freqs = []
                        st.warning("Cannot parse freq(s).")
                else:
                    freqs = []
                nr = notch_filter(st.session_state.raw_bandpassed, freqs)
                st.session_state.raw_notched = nr
                st.success(f"Notch done freq(s)={freqs}")
            show_basic_info_panel(4, "Notched", st.session_state.raw_notched, st.session_state.seizure_intervals)
            if st.session_state.raw_notched is not None:
                ro = st.session_state.raw_notched
                with st.expander("Notched Multi-Channel Overview", expanded=True):
                    all_chs = list(range(len(ro.ch_names)))
                    picks = st.multiselect("Channels (Notched)", all_chs, default=[0],
                                           format_func=lambda i: ro.ch_names[i])
                    off_f = st.number_input("Offset factor (Notch)", 50.0, 2000.0, 300.0, step=50.0)
                    if picks:
                        fignt = plot_entire_eeg_multichannel_two_color(
                            ro, picks,
                            red_intervals=st.session_state.seizure_intervals,
                            green_intervals=None,
                            offset_factor=off_f, decimate=True,
                            dynamic_offset=False, scale='uV',
                            show_rangeslider=False
                        )
                        st.plotly_chart(fignt, use_container_width=False, height=300)
                with st.expander("Compare PSD: Before & After Notch", expanded=True):
                    fn = plot_freq_domain_comparison(
                        st.session_state.raw_bandpassed, st.session_state.raw_notched,
                        "Before Notch", "After Notch"
                    )
                    st.pyplot(fn)

    # STEP 5: Normalize
    with tabs[4]:
        st.subheader("5. Normalize (No longer in µV)")
        if st.session_state.raw_notched is None:
            st.warning("Finish step 4.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Global Std"):
                    g = normalize_global_std(st.session_state.raw_notched)
                    st.session_state.raw_normalized = g
                    st.success("Global norm done.")
            with c2:
                if st.button("Channel-Wise"):
                    cwise = normalize_channel_wise(st.session_state.raw_notched)
                    st.session_state.raw_normalized = cwise
                    st.success("Channel-Wise norm done.")
            with c3:
                if st.button("Z-Score"):
                    z = normalize_zscore(st.session_state.raw_notched)
                    st.session_state.raw_normalized = z
                    st.success("Z-Score norm done.")
            show_basic_info_panel(5, "Normalized", st.session_state.raw_normalized, st.session_state.seizure_intervals)
            if st.session_state.raw_normalized is not None:
                ro = st.session_state.raw_normalized
                with st.expander("Normalized Multi-Channel Overview", expanded=True):
                    all_chs = list(range(len(ro.ch_names)))
                    # Default to one channel (index 0)
                    picks = st.multiselect("Channels (Normalized)", all_chs, default=[0],
                                           format_func=lambda i: ro.ch_names[i])
                    use_dyn = st.checkbox("Use Dynamic Offset?", value=True)
                    off_f = st.number_input("Offset factor (if NOT dynamic)", 1.0, 2000.0, 300.0, step=50.0)
                    decim = st.checkbox("Decimate Data in Plot?", value=True)
                    if picks:
                        fig_norm = plot_entire_eeg_multichannel_two_color(
                            ro, picks,
                            red_intervals=st.session_state.seizure_intervals,
                            green_intervals=None,
                            offset_factor=off_f, decimate=decim,
                            dynamic_offset=use_dyn, scale='norm',
                            show_rangeslider=False
                        )
                        st.plotly_chart(fig_norm, use_container_width=False, height=300)
                with st.expander("Normalized Snippet (Matplotlib)", expanded=True):
                    dur_ = st.slider("Snippet Dur (Norm)", 1.0, min(30.0, ro.times[-1]), 5.0)
                    stt_ = st.slider("Snippet Start (Norm)", 0.0, float(ro.times[-1]-dur_), 0.0)
                    if st.button("Plot Normalized Snippet"):
                        sfig = plot_snippet_matplotlib(
                            ro, stt_, dur_,
                            intervals_red=st.session_state.seizure_intervals,
                            intervals_green=None,
                            title="Normalized Snippet"
                        )
                        if sfig:
                            st.pyplot(sfig)

    # STEP 6: Predict
    with tabs[5]:
        st.subheader("6. Predict")
        if st.session_state.model is None:
            st.warning("No model loaded. Choose or load in sidebar.")
        elif st.session_state.raw_normalized is None:
            st.warning("Finish step 5 first.")
        else:
            raw_obj = st.session_state.raw_normalized
            total_dur = raw_obj.times[-1]

            # Panel 6.1: Predict All Windows with Progress Bar & Timing
            with st.expander("6.1 Predict All Windows", expanded=False):
                with st.form(key="predict_all_windows"):
                    window_sec = st.number_input("Window Size (s)", 1.0, 10.0, 3.0, step=0.5)
                    overlap_sec = st.number_input("Overlap (s)", 0.0, window_sec-0.5, 1.0, step=0.5)
                    submit1 = st.form_submit_button("Run Sliding-Window Prediction")
                    if submit1:
                        start_time_exec = time.time()
                        data = raw_obj.get_data()
                        total_samples = data.shape[1]
                        sfreq = raw_obj.info['sfreq']
                        w_samps = int(window_sec * sfreq)
                        step_samps = int((window_sec - overlap_sec) * sfreq)
                        total_windows = int(np.floor((total_samples - w_samps) / step_samps)) + 1

                        progress_bar = st.progress(0)
                        intervals = []
                        start_samp = 0
                        current_window = 0
                        model = st.session_state.model
                        model.eval()
                        while (start_samp + w_samps) <= total_samples:
                            seg_data = data[:, start_samp:(start_samp+w_samps)]
                            t0 = start_samp / sfreq
                            t1 = (start_samp + w_samps) / sfreq
                            seg_tensor = torch.tensor(seg_data, dtype=torch.float32).unsqueeze(0)
                            with torch.no_grad():
                                out = model(seg_tensor)
                                label = torch.argmax(out, dim=1).item()  # 0=nonseiz, 1=seizure
                            if label == 1:
                                intervals.append((t0, t1-t0, "seizure"))
                            else:
                                intervals.append((t0, t1-t0, "nonseiz"))
                            start_samp += step_samps
                            current_window += 1
                            progress_bar.progress(min(current_window / total_windows, 1.0))
                        end_time_exec = time.time()
                        running_time = end_time_exec - start_time_exec
                        st.session_state.window_preds = intervals
                        n_seiz = sum(1 for (_, _, lbl) in intervals if lbl=='seizure')
                        dur_seiz = sum(d for (_, d, lbl) in intervals if lbl=='seizure')
                        st.success(f"Finished predictions in {running_time:.2f} seconds! Raw seizure segments: {n_seiz}, total raw duration: {dur_seiz:.2f}s (overlap included). Total EEG duration: {raw_obj.times[-1]:.2f}s.")

            # Panel 6.3: Post-process & Final Plotly with Channel Selection
            with st.expander("6.3 Post-process & Final Plotly", expanded=False):
                with st.form(key="merge_complement"):
                    mgap = st.number_input("Merge Gap (s)", 0.0, 10.0, 2.0, 0.5)
                    submit3 = st.form_submit_button("Merge & Complement")
                    if submit3:
                        raw_seiz_list = [(s, d) for (s, d, l) in st.session_state.window_preds if l=='seizure']
                        merges = merge_intervals(raw_seiz_list, mgap)
                        # Remove predicted seizure intervals shorter than 20s
                        merges_filtered = [(s, dur) for (s, dur) in merges if dur >= 25]
                        st.session_state.pred_seizure_intervals = merges_filtered
                        st.session_state.pred_nonseizure_intervals = complement_intervals(total_dur, merges_filtered)
                        st.success("Merged intervals done (filtered <20s)!")
                        seg_count = len(st.session_state.pred_seizure_intervals)
                        dur_s = sum(d for (_, d) in st.session_state.pred_seizure_intervals)
                        st.write(f"Merged seizure segments (>=20s): {seg_count}, total merged duration: {dur_s:.2f}s.")
                        show_predict_info_sidebar()

                # Final figure with truth label (red) and predicted label (green) with channel selection
                if st.session_state.pred_seizure_intervals:
                    col1, col2 = st.columns([2,1])
                    with col1:
                        all_chs = list(range(len(raw_obj.ch_names)))
                        # Default to one channel (index 0) for channel selection in step 6.3
                        picks = st.multiselect("Select Channels", all_chs, default=[0],
                                                 format_func=lambda i: raw_obj.ch_names[i])
                    with col2:
                        dec = st.checkbox("Decimate final plot?", value=False)
                    channels_to_plot = picks if picks else all_chs
                    fig_final = plot_entire_eeg_multichannel_two_color(
                        raw_obj, channels_to_plot,
                        red_intervals=st.session_state.seizure_intervals,  # ground truth in red
                        green_intervals=st.session_state.pred_seizure_intervals,  # predicted in green
                        offset_factor=300.0,
                        decimate=dec,
                        dynamic_offset=True,
                        scale='norm',
                        force_predict_range=False,
                        show_rangeslider=False  # remove bottom bar in 6.3
                    )
                    # Use HTML to color the legend text in the same colors as the plot
                    st.write('<span style="color: rgb(255,0,0);">Red = ground truth seizure</span>, <span style="color: rgb(0,255,0);">Green = predicted seizure</span>', unsafe_allow_html=True)
                    st.plotly_chart(fig_final, use_container_width=True, height=600, config={"toImageButtonOptions": {"width": 1920, "height": 1080}})
                else:
                    st.info("No final intervals yet. Merge in 6.3.")

            # Panel 6.4: Results Table in Two Parts (Ground Truth & Prediction)
            with st.expander("6.4 Results Table", expanded=True):
                if st.session_state.pred_seizure_intervals:
                    df_gt = pd.DataFrame(st.session_state.seizure_intervals, columns=["Onset (s)", "Duration (s)"]).round(1)
                    df_pred = pd.DataFrame(st.session_state.pred_seizure_intervals, columns=["Onset (s)", "Duration (s)"]).round(1)
                    col_gt, col_pred = st.columns(2)
                    with col_gt:
                        st.write("**Ground Truth Seizure Intervals**")
                        st.table(df_gt)
                    with col_pred:
                        st.write("**Predicted Seizure Intervals**")
                        st.table(df_pred)
                    # Compute prediction accuracy as the total overlapping duration between ground truth and prediction divided by total ground truth seizure duration
                    gt_total = sum(d for (_, d) in st.session_state.seizure_intervals)
                    overlap = compute_overlap_duration(st.session_state.seizure_intervals, st.session_state.pred_seizure_intervals)
                    accuracy = overlap / gt_total * 100 if gt_total > 0 else 0
                    st.write(f"**Prediction Accuracy (Overlap / Ground Truth): {accuracy:.2f}%**")
                else:
                    st.info("No final intervals yet. Merge in 6.3.")

if __name__ == "__main__":
    main()
