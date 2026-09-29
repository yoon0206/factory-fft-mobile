
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks
import io

st.set_page_config(page_title="Factory FFT Analyzer", page_icon="📊", layout="wide")
st.title("Factory Noise / FFT Analyzer")
st.caption("Mobile-friendly frequency analysis for machine noise and vibration recordings")

rpm = st.number_input("Machine RPM", min_value=0.0, value=1500.0, step=10.0)
rot_hz = rpm / 60.0 if rpm > 0 else 0
st.metric("Rotational frequency", f"{rot_hz:.2f} Hz")

uploaded = st.file_uploader("Upload WAV recording", type=["wav"])

if uploaded:
    fs, y = wavfile.read(io.BytesIO(uploaded.read()))
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(float)
    y -= np.mean(y)

    # Limit analysis length for smooth mobile use.
    max_samples = int(fs * 60)
    y = y[:max_samples]

    n = len(y)
    window = np.hanning(n)
    spectrum = np.fft.rfft(y * window)
    freq = np.fft.rfftfreq(n, 1/fs)
    amp = np.abs(spectrum) * 2 / max(np.sum(window), 1)

    # Display range.
    max_hz = min(10000.0, fs / 2)
    display_max = st.slider(
        "Maximum displayed frequency (Hz)",
        min_value=100,
        max_value=int(max_hz),
        value=min(5000, int(max_hz)),
        step=100,
    )
    mask = (freq > 0) & (freq <= display_max)
    f = freq[mask]
    a = amp[mask]

    if len(a):
        prominence = max(np.max(a) * 0.03, 1e-12)
        peaks, _ = find_peaks(a, prominence=prominence)
        order = peaks[np.argsort(a[peaks])[::-1]][:10] if len(peaks) else np.array([], dtype=int)

        st.subheader("FFT spectrum")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(f, a, linewidth=1)
        if rpm > 0 and rot_hz <= display_max:
            ax.axvline(rot_hz, linestyle="--", linewidth=1.5, label=f"1× RPM = {rot_hz:.2f} Hz")
            for k in range(2, 6):
                if k * rot_hz <= display_max:
                    ax.axvline(k * rot_hz, linestyle=":", linewidth=0.8)
            ax.legend()
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, use_container_width=True)

        st.subheader("Main frequency peaks")
        if len(order):
            rows = []
            for rank, idx in enumerate(order, 1):
                hz = float(f[idx])
                multiple = hz / rot_hz if rot_hz > 0 else np.nan
                rows.append({
                    "Rank": rank,
                    "Frequency (Hz)": round(hz, 2),
                    "Amplitude": round(float(a[idx]), 6),
                    "RPM multiple": round(multiple, 2) if rpm > 0 else "-"
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No clear frequency peak was detected.")

        if rpm > 0:
            st.info(
                f"At {rpm:.0f} RPM, 1× rotational frequency is {rot_hz:.2f} Hz. "
                "Compare strong peaks near 1×, 2×, 3×… with machine operating conditions."
            )
else:
    st.info("Record machine sound as WAV and upload it here. FFT analysis starts automatically.")
