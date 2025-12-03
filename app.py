"""
TimberMind V1 – Integrated App
------------------------------

This file glues together everyone’s work:

- TEJAS  : Microphone recording (live audio -> WAV file)
- AIDAN  : Source separation model (mixed WAV -> 2 stem WAVs via NMF)
- NICO   : Feature extraction + Perlin-noise visualizer
- JAN    : Tkinter user interface (buttons, tabs, and wiring)

High-level flow:

1. User picks an audio source:
   - Upload an existing .wav
   - Record from a microphone device

2. The chosen .wav is passed into Aidan’s separation code.
   That produces two new .wav files:
      *_source1.wav  -> we label as "Violin"
      *_source2.wav  -> we label as "Viola"

3. Nico computes time-varying audio features (RMS loudness and spectral
   centroid) for:
   - the original mix
   - the violin stem
   - the viola stem

4. Nico’s Perlin field uses those features to control:
   - stroke thickness      (RMS -> louder = thicker lines)
   - stroke length         (centroid -> brighter tone = longer strokes)
   - stroke color (blend)  (centroid -> smoothly blends between two colors)

5. Jan’s UI shows 3 tabs:
   - Original mix
   - Violin
   - Viola

   Each tab:
   - Plays its own audio (via sounddevice) when visible and “Play” is active
   - Stops audio when you switch away
   - Has its own Perlin field, revealed left-to-right as time advances
"""

# -------------------- STANDARD IMPORTS --------------------  # (shared)
import os
import time
import math
import random
import importlib.util
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox

# Optional audio / DSP libraries
try:
    import sounddevice as sd  # Used by Tejas (recording) and playback
except Exception:
    sd = None

sf = None
if importlib.util.find_spec("soundfile"):
    import soundfile as sf  # type: ignore

from scipy.io.wavfile import write as wav_write

# Aidan’s model dependencies
from sklearn.decomposition import NMF
import scipy.signal as signal
import matplotlib.pyplot as plt  # kept for completeness with original code (not used here)


# -----------------------------------------------------------
# SMALL HELPER (shared)
# -----------------------------------------------------------

def mmss(sec: float) -> str:
    """Format a number of seconds as MM:SS for the UI."""
    sec = int(max(0, sec or 0))
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"


# -----------------------------------------------------------
# EVENT BUS + APP STATE  (shared infrastructure)
# -----------------------------------------------------------

class EventBus:
    """
    Simple publish/subscribe system so the UI, audio, and visuals
    can talk to each other without being tightly coupled.
    """

    def __init__(self):
        self._subs = {}

    def subscribe(self, name, fn):
        self._subs.setdefault(name, []).append(fn)

    def dispatch(self, name, detail=None):
        for fn in self._subs.get(name, []):
            try:
                fn(detail)
            except Exception as e:
                print(f"[EVENT ERROR] {name}: {e}")


class AppState:
    """
    Global state shared across components.
    """

    def __init__(self):
        self.mode = "perlin"      # current visualization mode
        self.status = "stopped"   # "playing" or "paused"
        self.duration = 0.0       # seconds
        self.current = 0.0        # current time position
        self.active_view = "mix"  # which tab is active: "mix", "violin", "viola"


# =====================================================================
# TEJAS – MICROPHONE RECORDING (mic -> WAV file)
# =====================================================================

def record_audio_tejas(
    duration=5,
    sample_rate=44100,
    channels=1,
    device=None,
):
    """
    TEJAS SECTION
    -------------
    Records live audio from the selected microphone and converts it
    to a digital numpy array. Later we save this as a .wav file.

    Parameters:
        duration (int): seconds of recording
        sample_rate (int): samples per second
        channels (int): mono=1, stereo=2
        device (int or None): sounddevice device index; None = default

    Returns:
        digital_signal (np.ndarray): 1D array with audio samples
        sample_rate (int): sampling rate used
    """
    if sd is None:
        raise RuntimeError(
            "sounddevice is not available. Install it with 'pip install sounddevice'."
        )

    print("Recording audio... (speak / play now)")
    audio_data = sd.rec(
        int(sample_rate * duration),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    print("Recording complete!")

    # Clip to [-1, 1] just in case and flatten to 1D (mono-like)
    audio_data = np.clip(audio_data, -1, 1)
    digital_signal = audio_data.flatten()

    return digital_signal, sample_rate


def save_recording_to_wav(signal_arr, fs, out_path):
    """
    Small helper that converts Tejas's float32 signal into int16
    and writes a WAV file on disk.
    """
    wav_data = np.int16(signal_arr * 32767)
    wav_write(out_path, fs, wav_data)
    return out_path


# =====================================================================
# AIDAN – SOURCE SEPARATION MODEL (mixed WAV -> 2 stem WAVs)
# =====================================================================

"""
IMPORTANT: For this section we keep Aidan's machine learning code
as-is. We only *call* these functions from the rest of the app.
"""

def load_audio_mono(path):
    """
    AIDAN SECTION
    -------------
    Load audio with soundfile and convert to mono float32 in [-1, 1].
    """
    y, sr = sf.read(path)  # y: [N] or [N, C]

    # Stereo -> mono
    if y.ndim == 2:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    return y, sr


def stft_mag_phase(y, sr, n_fft=2048, hop_length=512):
    """
    Compute STFT using scipy.signal.stft and return magnitude + phase.
    hop_length = n_fft - noverlap
    """
    nperseg = n_fft
    noverlap = n_fft - hop_length

    f, t, Zxx = signal.stft(
        y,
        fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
    )
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    return mag, phase, f, t


def istft_from_mag_phase(mag, phase, sr, n_fft=2048, hop_length=512):
    """
    Inverse STFT using scipy.signal.istft from magnitude and phase.
    """
    nperseg = n_fft
    noverlap = n_fft - hop_length
    Zxx = mag * np.exp(1j * phase)

    _, y = signal.istft(
        Zxx,
        fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
    )
    return y.astype(np.float32)


def nmf_separate(mag, n_components=2, max_iter=200):
    """
    Non-negative matrix factorization.
    Returns list of magnitude estimates, one per source.
    mag: [F, T]
    """
    F, T = mag.shape
    X = mag + 1e-8  # avoid zeros

    model = NMF(
        n_components=n_components,
        init="random",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=max_iter,
        random_state=0,
    )
    W = model.fit_transform(X)  # [F, K]
    H = model.components_       # [K, T]

    mags = []
    for k in range(n_components):
        Wk = W[:, [k]]   # [F, 1]
        Hk = H[[k], :]   # [1, T]
        Mk = np.dot(Wk, Hk)  # [F, T]
        mags.append(Mk)
    return mags


def separate_file(input_path, output_prefix):
    """
    AIDAN SECTION (kept intact)
    ---------------------------
    This function takes an input WAV file, performs NMF-based
    source separation, and writes two stem WAV files:

        {output_prefix}_source1.wav
        {output_prefix}_source2.wav

    In our app we *interpret* source1 as "Violin" and source2 as "Viola"
    purely for visualization / labeling. The ML itself is unchanged.
    """
    print(f"Loading {input_path}...")
    y, sr = load_audio_mono(input_path)

    print("Computing STFT...")
    n_fft = 2048
    hop = 512
    mag, phase, f, t = stft_mag_phase(y, sr, n_fft=n_fft, hop_length=hop)

    print("Performing NMF separation...")
    mags = nmf_separate(mag, n_components=2, max_iter=300)

    print("Reconstructing time-domain signals...")
    eps = 1e-10
    mag_sum = mags[0] + mags[1] + eps

    stem_paths = []

    for i, mag_i in enumerate(mags):
        # Soft mask to keep consistent with original mix energy
        mask = mag_i / mag_sum
        mag_masked = mag * mask

        yi = istft_from_mag_phase(mag_masked, phase, sr, n_fft=n_fft, hop_length=hop)

        out_path = f"{output_prefix}_source{i+1}.wav"
        # soundfile will write float32 WAV by default
        sf.write(out_path, yi, sr, subtype="PCM_16")
        print(f"Saved: {out_path}")
        stem_paths.append(out_path)

    print("Done.")
    return stem_paths, sr


# =====================================================================
# NICO – FEATURE EXTRACTION + PERLIN VISUALIZER
# =====================================================================

# ---- FeatureBank: time-varying audio descriptors for each view -------

class FeatureBank:
    """
    NICO SECTION
    ------------
    Holds per-frame audio features for:
        - "mix":   original recording
        - "violin": separated stem 1
        - "viola":  separated stem 2

    For each view we store:
        times[t]        : time stamp of frame t
        rms[t]          : loudness (root-mean-square amplitude)
        centroid[t]     : spectral centroid (brightness)
    """

    def __init__(self):
        self.data = {}
        self.max_rms = 1e-6
        self.max_centroid = 1e-6

    def compute_for_signal(self, key, signal_arr, sample_rate, win_size=1024, hop=512):
        """
        Slice the signal into short overlapping windows and compute:

        - magnitude spectrogram
        - per-frame RMS loudness
        - per-frame spectral centroid
        """
        if signal_arr is None or len(signal_arr) == 0:
            self.data[key] = {
                "times": np.array([0.0]),
                "rms": np.array([0.0]),
                "centroid": np.array([0.0]),
            }
            return

        frames = 1 + max(0, (len(signal_arr) - win_size) // hop)
        window = np.hanning(win_size)

        mag_spec = np.zeros((win_size // 2 + 1, frames), dtype=float)
        rms_vals = np.zeros(frames, dtype=float)

        for i in range(frames):
            start = i * hop
            frame = signal_arr[start: start + win_size]
            if len(frame) < win_size:
                frame = np.pad(frame, (0, win_size - len(frame)))
            frame = frame * window

            # RMS loudness in this frame
            rms_vals[i] = math.sqrt(float(np.mean(frame**2)) + 1e-12)

            # Spectrum for centroid
            spectrum = np.fft.rfft(frame)
            mag_spec[:, i] = np.abs(spectrum)

        freqs = np.fft.rfftfreq(win_size, 1.0 / sample_rate)
        times = np.arange(frames) * hop / float(sample_rate)

        # Spectral centroid: weighted average frequency
        centroid_num = np.sum(mag_spec * freqs[:, None], axis=0)
        centroid_den = np.sum(mag_spec + 1e-12, axis=0)
        centroid = centroid_num / centroid_den

        self.data[key] = {
            "times": times,
            "rms": rms_vals,
            "centroid": centroid,
        }

        self.max_rms = max(self.max_rms, float(np.max(rms_vals)))
        self.max_centroid = max(self.max_centroid, float(np.max(centroid)))

    def get_features(self, key, t):
        """
        Get interpolated features for a given time t (seconds) and view key.
        If something is missing, return zeros to keep the visual stable.
        """
        entry = self.data.get(key)
        if entry is None:
            return {"rms": 0.0, "centroid": 0.0}

        times = entry["times"]
        if len(times) == 0:
            return {"rms": 0.0, "centroid": 0.0}

        t = float(max(0.0, min(times[-1], t)))
        rms = float(np.interp(t, times, entry["rms"]))
        centroid = float(np.interp(t, times, entry["centroid"]))
        # Normalize so visuals are scale-independent
        rms_norm = min(1.0, rms / self.max_rms) if self.max_rms > 0 else 0.0
        cent_norm = min(1.0, centroid / self.max_centroid) if self.max_centroid > 0 else 0.0
        return {
            "rms": rms_norm,
            "centroid": cent_norm,
        }


# ---- Color helpers for Nico’s Perlin mapping -------------------------

def _hex_to_rgb(color):
    color = color.lstrip("#")
    return (
        int(color[0:2], 16),
        int(color[2:4], 16),
        int(color[4:6], 16),
    )


def _rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def _color_lerp(c1, c2, alpha):
    """
    Linearly blend two hex colors c1 and c2 with factor alpha in [0,1].
    alpha=0 -> c1, alpha=1 -> c2.
    """
    alpha = max(0.0, min(1.0, alpha))
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    r = int(r1 + (r2 - r1) * alpha)
    g = int(g1 + (g2 - g1) * alpha)
    b = int(b1 + (b2 - b1) * alpha)
    return _rgb_to_hex(r, g, b)


class PerlinVisualizer:
    """
    NICO SECTION
    ------------
    Responsible for drawing the Perlin-noise flow fields for all three
    views ("mix", "violin", "viola") onto their respective canvases.

    The field is built *offline* into a list of segments. During playback
    we simply reveal more segments from left to right based on the current
    time position.

    How features control the visual:

    - rms (loudness)
        -> line thickness (louder = thicker lines)
    - centroid (brightness)
        -> stroke length (higher centroid = longer strokes)
        -> color blend (low centroid = base color A, high = color B)
    - time t
        -> horizontal position (left = early; right = late)
    """

    def __init__(self, bus, state, feature_bank,
                 canvas_mix: tk.Canvas, canvas_violin: tk.Canvas, canvas_viola: tk.Canvas):
        self.bus = bus
        self.state = state
        self.feature_bank = feature_bank
        self.canvases = {
            "mix": canvas_mix,
            "violin": canvas_violin,
            "viola": canvas_viola,
        }
        # For scheduling redraws
        self.canvas = canvas_mix

        # Base color pairs for each view (low centroid -> first, high -> second)
        self.color_pairs = {
            "mix": ("#4c1d95", "#f97316"),      # deep purple -> bright orange
            "violin": ("#7c3aed", "#fb7185"),   # violet -> pink
            "viola": ("#0ea5e9", "#a3e635"),    # teal -> lime
        }

        # Perlin parameters
        self.grid_size = 96
        self.noise_scale_x = 220.0
        self.noise_scale_y = 220.0
        self.flow_speed = 0.05
        self.gradients = self._generate_gradients()

        # Precomputed segments per view: list of (xmax, x0, y0, x1, y1, color, width)
        self.segments = {"mix": [], "violin": [], "viola": []}
        self.segment_indices = {"mix": 0, "violin": 0, "viola": 0}
        self.drawn_until = {"mix": 0.0, "violin": 0.0, "viola": 0.0}
        self._build_job = None

        bus.subscribe("app:newAudio", self._on_new_audio)
        bus.subscribe("app:setActiveView", self._on_active_view)

    # ----------- event handlers ---------------------------------------

    def _on_new_audio(self, _detail=None):
        # New recording or uploaded file: rebuild all Perlin fields
        self.reset(wipe_segments=True)
        self._schedule_build()

    def _on_active_view(self, detail):
        # When the user switches tabs, we simply store the new active view.
        if detail and "view" in detail:
            self.state.active_view = detail["view"]

    # ----------- reset / build management ------------------------------

    def reset(self, wipe_segments=True):
        """Clear canvases and optionally discard the pre-built segments."""
        for canvas in self.canvases.values():
            canvas.delete("all")
        self.drawn_until = {k: 0.0 for k in self.drawn_until}
        self.segment_indices = {k: 0 for k in self.segment_indices}
        if wipe_segments:
            self.segments = {"mix": [], "violin": [], "viola": []}

    def _schedule_build(self):
        """Schedule a background build once the canvases know their size."""
        if self._build_job:
            self.canvas.after_cancel(self._build_job)
        self._build_job = self.canvas.after(80, self._build_segments)

    # ----------- Perlin core ------------------------------------------

    def _generate_gradients(self):
        gradients = {}
        rng = random.Random(11)
        for gx in range(self.grid_size + 1):
            for gy in range(self.grid_size + 1):
                angle = rng.random() * math.tau
                gradients[(gx, gy)] = (math.cos(angle), math.sin(angle))
        return gradients

    def _fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _dot_grid_gradient(self, ix, iy, x, y):
        gradient = self.gradients.get((ix, iy), (0.0, 0.0))
        dx = x - ix
        dy = y - iy
        return dx * gradient[0] + dy * gradient[1]

    def _perlin(self, x, y):
        x0 = int(x)
        x1 = x0 + 1
        y0 = int(y)
        y1 = y0 + 1
        sx = self._fade(x - x0)
        sy = self._fade(y - y0)
        n0 = self._dot_grid_gradient(x0, y0, x, y)
        n1 = self._dot_grid_gradient(x1, y0, x, y)
        ix0 = n0 + sx * (n1 - n0)
        n0 = self._dot_grid_gradient(x0, y1, x, y)
        n1 = self._dot_grid_gradient(x1, y1, x, y)
        ix1 = n0 + sx * (n1 - n0)
        return ix0 + sy * (ix1 - ix0)

    # ----------- building segments ------------------------------------

    def _build_segments(self):
        """
        Precompute trajectories for all three views.
        We do this once per audio clip so playback is cheap.
        """
        self._build_job = None
        self.gradients = self._generate_gradients()

        # We use the mix canvas as reference for sizing
        base_canvas = self.canvases["mix"]
        width = max(400, base_canvas.winfo_width() or 1200)
        height = max(300, base_canvas.winfo_height() or 700)
        duration = max(self.state.duration, 1.0)

        for view in ["mix", "violin", "viola"]:
            c1, c2 = self.color_pairs[view]
            self.segments[view] = self._generate_segments_for_view(
                view, c1, c2, width, height, duration
            )
            self.segments[view].sort(key=lambda seg: seg[0])

        self.segment_indices = {k: 0 for k in self.segment_indices}
        self.drawn_until = {k: 0.0 for k in self.drawn_until}

    def _generate_segments_for_view(self, view_key, color_low, color_high, width, height, duration):
        """
        Generate a bunch of Perlin-guided strokes for a given view
        ("mix", "violin", or "viola").

        Particles:
            - start at random positions
            - move according to the Perlin field
            - stroke properties depend on features at the corresponding time

        Mapping:
            - RMS -> thickness
            - centroid -> step length + color blend between color_low and color_high
        """
        # For variety between instruments we use view-specific seeds
        seed_map = {"mix": 111, "violin": 123, "viola": 321}
        rng = random.Random(seed_map.get(view_key, 0))

        particles = 340          # more = denser image, but heavier CPU
        steps = 150              # more = longer trails

        segments = []

        for _ in range(particles):
            x = rng.random() * width
            y = rng.random() * height
            for _ in range(steps):
                # Map x-position into a time index along the clip
                t = (x / float(width)) * duration
                feats = self.feature_bank.get_features(view_key, t)
                rms = feats["rms"]
                centroid = feats["centroid"]

                # Map features -> visual properties
                #   - louder = thicker lines
                #   - brighter tone = longer steps and more towards high color
                thickness = 0.4 + rms * 5.0
                step_len = 0.8 + centroid * 6.0

                color = _color_lerp(color_low, color_high, centroid)

                # Perlin noise controls direction of travel
                nx = (x / self.noise_scale_x) + t * self.flow_speed
                ny = (y / self.noise_scale_y) + t * self.flow_speed
                angle = self._perlin(nx % self.grid_size, ny % self.grid_size) * math.tau

                x1 = x + math.cos(angle) * step_len
                y1 = y + math.sin(angle) * step_len
                y1 = max(0.0, min(height, y1))

                # If particle escapes the canvas, respawn it
                if x1 < -5 or x1 > width + 5:
                    x = rng.random() * width
                    y = rng.random() * height
                    continue

                seg_max = max(x, x1)
                segments.append((seg_max, x, y, x1, y1, color, thickness))

                x, y = x1, y1

        return segments

    # ----------- rendering ---------------------------------------------

    def render_frame(self, current_time: float):
        """
        Called regularly by the playback loop.

        Only the *active* tab’s canvas is drawn to keep performance high.
        We reveal strokes left-to-right according to the current playback time.
        """
        if self.state.mode != "perlin":
            # In the future you could add other visualization modes here.
            return

        if not any(self.segments.values()):
            # First time through, build the segments
            if self._build_job is None:
                self._schedule_build()
            return

        duration = max(self.state.duration, 1e-6)
        progress = max(0.0, min(1.0, current_time / duration))

        active_view = self.state.active_view
        canvas = self.canvases.get(active_view)
        if canvas is None:
            return

        width = max(1, canvas.winfo_width())
        x_limit = progress * width

        idx = self.segment_indices.get(active_view, 0)
        segments = self.segments.get(active_view, [])

        while idx < len(segments) and segments[idx][0] <= x_limit:
            _, x0, y0, x1, y1, color, w = segments[idx]
            canvas.create_line(
                x0, y0, x1, y1,
                fill=color,
                width=w,
                smooth=True,
                capstyle=tk.ROUND,
            )
            idx += 1

        self.segment_indices[active_view] = idx
        self.drawn_until[active_view] = x_limit

    # Public helper used by the Reset button
    def clear_canvases(self):
        self.reset(wipe_segments=False)


# =====================================================================
# AUDIO CONTROLLER (shared; Nico + integration glue)
# =====================================================================

class AudioController:
    """
    Manages audio data for all three views and drives the global
    playback clock used by the visualizer.

    NEW: Uses sounddevice to actually play back the audio:

    - Only the currently active tab's audio is audible.
    - Switching tabs while playing restarts playback for that view
      at the *same* time position.
    """

    def __init__(self, bus, state):
        self.bus = bus
        self.state = state

        # Each entry is a 1D numpy array representing the waveform
        self.signals = {
            "mix": None,
            "violin": None,
            "viola": None,
        }
        self.sample_rate = 44100
        self.duration = 0.0
        self.current_time = 0.0
        self.playing = False
        self.current_view_playing = None  # which view's audio is currently sent to sounddevice

        bus.subscribe("app:setActiveView", self._on_active_view)

    def _on_active_view(self, detail):
        """
        When the user changes the visible tab, we either:
        - start playback for the new view (if playing),
        - or make sure all audio is stopped.
        """
        if detail and "view" in detail:
            self.state.active_view = detail["view"]
        if self.playing:
            self._start_playback_for_active_view()
        else:
            self._stop_audio()

    def set_signals(self, mix, violin, viola, sr):
        """
        Called once after separation is complete.
        """
        self.signals["mix"] = mix
        self.signals["violin"] = violin
        self.signals["viola"] = viola
        self.sample_rate = sr

        # Use the mix length as the time base
        self.duration = len(mix) / float(sr) if mix is not None else 0.0
        self.current_time = 0.0
        self.state.duration = self.duration
        self.state.current = 0.0

        self.playing = False
        self.current_view_playing = None
        self._stop_audio()

        self.bus.dispatch("media:state", {
            "status": "paused",
            "duration": self.duration,
            "currentTime": self.current_time,
        })

    def _start_playback_for_active_view(self):
        """
        Start audio playback for the currently active tab from the
        current time position, using sounddevice.
        """
        if sd is None:
            # If sounddevice is not available, we silently skip playback.
            self.current_view_playing = None
            return

        view = self.state.active_view
        sig = self.signals.get(view)
        if sig is None:
            self.current_view_playing = None
            sd.stop()
            return

        # Starting index according to current_time
        start_idx = int(max(0.0, min(self.current_time, self.duration)) * self.sample_rate)
        start_idx = min(len(sig), start_idx)
        segment = sig[start_idx:]

        # Stop any previous playback and start this one
        sd.stop()
        if len(segment) > 0:
            sd.play(segment, samplerate=self.sample_rate, blocking=False)
            self.current_view_playing = view
        else:
            self.current_view_playing = None

    def _stop_audio(self):
        """Stop any ongoing playback."""
        if sd is not None:
            sd.stop()
        self.current_view_playing = None

    def step(self, dt):
        """
        Advance the playback clock by dt seconds when playing.
        """
        if not self.playing:
            return self.current_time

        self.current_time += dt
        if self.duration > 0 and self.current_time >= self.duration:
            # Reached end: stop both animation and audio
            self.current_time = self.duration
            self.playing = False
            self._stop_audio()

        self.state.current = self.current_time
        return self.current_time

    def play(self):
        """Called when user hits Play."""
        if self.duration <= 0:
            return
        self.playing = True
        self.current_time = 0.0
        self.state.status = "playing"
        self._start_playback_for_active_view()

    def pause(self):
        """Called when user hits Pause."""
        self.playing = False
        self.state.status = "paused"
        self._stop_audio()


# =====================================================================
# PLAYBACK LOOP (shared)
# =====================================================================

class PlaybackLoop:
    """
    Small helper that periodically steps the audio clock and asks
    the visualizer to draw a new frame.
    """

    def __init__(self, bus, state, audio_ctrl, visualizer: PerlinVisualizer):
        self.bus = bus
        self.state = state
        self.audio_ctrl = audio_ctrl
        self.visualizer = visualizer
        self.running = False
        self.last_time = None

        bus.subscribe("app:play", self._start)
        bus.subscribe("app:pause", self._stop)

    def _start(self, _detail=None):
        if self.running:
            return
        self.running = True
        self.last_time = time.time()
        self._tick()

    def _stop(self, _detail=None):
        self.running = False

    def _tick(self):
        if not self.running:
            return
        now = time.time()
        dt = now - (self.last_time or now)
        self.last_time = now

        current_time = self.audio_ctrl.step(dt)
        self.visualizer.render_frame(current_time)
        self.bus.dispatch("media:state", {
            "status": "playing" if self.audio_ctrl.playing else "paused",
            "duration": self.audio_ctrl.duration,
            "currentTime": current_time,
        })

        # Target ~60fps
        self.visualizer.canvas.after(16, self._tick)


# =====================================================================
# JAN – TKINTER USER INTERFACE
# =====================================================================

class VisualizerApp(tk.Tk):
    """
    JAN SECTION
    -----------
    Tkinter window containing:
    - Device selection + recording controls
    - Upload button
    - Reset button
    - Mode selector (for future extension)
    - Tabs for Mix / Violin / Viola Perlin visuals

    New in this version:
    - Very obvious "Record from Mic" button in the top bar
    - Slightly more modern styling (bigger title, spacing, muted text)
    """

    def __init__(self, bus, state, feature_bank, audio_ctrl, visualizer):
        super().__init__()
        self.bus = bus
        self.state = state
        self.feature_bank = feature_bank
        self.audio_ctrl = audio_ctrl
        self.visualizer = visualizer

        self.title("TimberMind – Instrument Visualizer")
        self.geometry("1180x740")
        self.minsize(960, 580)

        # Core colors for a dark, modern look.
        self.c_bg = "#020617"
        self.c_panel = "#020617"
        self.c_card = "#020617"
        self.c_line = "#1e293b"
        self.c_accent = "#6366f1"
        self.c_text = "#e5e7eb"
        self.c_muted = "#64748b"

        self.mode_var = tk.StringVar(value="perlin")

        self._make_style()
        self._build_layout()

        self.bus.subscribe("media:state", self._on_media_state)
        self._refresh_time()

    # ---------- style & layout ----------------------------------------

    def _make_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=self.c_bg)
        style.configure("TLabel", background=self.c_bg, foreground=self.c_text, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=self.c_bg, foreground=self.c_muted, font=("Segoe UI", 9))
        style.configure(
            "Primary.TButton",
            background=self.c_accent,
            foreground="#0f172a",
            borderwidth=0,
            padding=8,
            focusthickness=0,
        )
        style.map("Primary.TButton", background=[("active", "#4f46e5")])
        style.configure(
            "Ghost.TButton",
            background="#0f172a",
            foreground=self.c_text,
            borderwidth=1,
            padding=8,
            focusthickness=0,
            relief="flat",
        )
        style.map("Ghost.TButton", background=[("active", "#111827")])
        style.configure("Card.TFrame", background="#020617")
        style.configure("Card.TLabelframe", background="#020617", foreground=self.c_text)
        style.configure("Card.TLabelframe.Label", background="#020617", foreground=self.c_muted)
        style.configure("Time.TLabel", background=self.c_bg, foreground=self.c_muted, font=("Segoe UI", 10, "bold"))

    def _build_layout(self):
        # Top bar
        top = ttk.Frame(self, padding=(20, 16))
        top.pack(side="top", fill="x")

        # Left: title and subtitle
        title_box = ttk.Frame(top, style="TFrame")
        title_box.pack(side="left", anchor="w")
        ttk.Label(
            title_box,
            text="TimberMind",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            title_box,
            text="Violin & Viola timbre visualizer – mix, separate, and *see* the sound.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        # Right: transport + recording controls
        right_box = ttk.Frame(top, style="TFrame")
        right_box.pack(side="right", anchor="e")

        # Big Record button (very visible)
        ttk.Button(
            right_box,
            text="● Record from Mic",
            style="Primary.TButton",
            command=self._on_record,
        ).pack(side="right", padx=(8, 0))

        ttk.Button(
            right_box,
            text="Upload .wav",
            style="Ghost.TButton",
            command=self._on_upload,
        ).pack(side="right", padx=8)

        ttk.Button(
            right_box,
            text="Reset Canvas",
            style="Ghost.TButton",
            command=self._on_reset,
        ).pack(side="right", padx=8)

        # Play / Pause + time
        transport_box = ttk.Frame(right_box, style="TFrame")
        transport_box.pack(side="right", padx=(0, 8))
        ttk.Button(transport_box, text="▶ Play", style="Ghost.TButton", command=self._on_play).pack(side="left", padx=(0, 4))
        ttk.Button(transport_box, text="⏸ Pause", style="Ghost.TButton", command=self._on_pause).pack(side="left", padx=(0, 4))
        self.time_lbl = ttk.Label(transport_box, text="00:00 / 00:00", style="Time.TLabel")
        self.time_lbl.pack(side="left", padx=(6, 0))

        # Body: left controls, right canvases
        body = ttk.Frame(self, padding=(20, 0, 20, 20))
        body.pack(fill="both", expand=True)

        controls = ttk.Frame(body, width=320)
        controls.pack(side="left", fill="y", padx=(0, 16))

        self._input_section(controls)
        self._mode_section(controls)

        # Right: notebook with three canvases
        canvas_frame = ttk.Frame(body, style="Card.TFrame")
        canvas_frame.pack(side="right", fill="both", expand=True)
        canvas_frame.pack_propagate(False)

        self.notebook = ttk.Notebook(canvas_frame)
        self.notebook.pack(fill="both", expand=True, padx=4, pady=4)

        self.canvas_mix = tk.Canvas(self.notebook, bg="#020617", highlightthickness=1, highlightbackground=self.c_line)
        self.canvas_violin = tk.Canvas(self.notebook, bg="#020617", highlightthickness=1, highlightbackground=self.c_line)
        self.canvas_viola = tk.Canvas(self.notebook, bg="#020617", highlightthickness=1, highlightbackground=self.c_line)

        self.notebook.add(self.canvas_mix, text="Original mix")
        self.notebook.add(self.canvas_violin, text="Violin")
        self.notebook.add(self.canvas_viola, text="Viola")

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

        # Attach the real canvases to the visualizer now
        self.visualizer.canvases["mix"] = self.canvas_mix
        self.visualizer.canvases["violin"] = self.canvas_violin
        self.visualizer.canvases["viola"] = self.canvas_viola
        self.visualizer.canvas = self.canvas_mix

    # ---------- left panel sections -----------------------------------

    def _input_section(self, parent):
        box = ttk.Labelframe(parent, text="Input / Analysis", style="Card.TLabelframe")
        box.pack(fill="x", pady=8)

        ttk.Label(
            box,
            text="Choose how to provide audio (Jan UI):",
            style="Muted.TLabel",
        ).pack(anchor="w", padx=10, pady=(8, 2))

        ttk.Label(
            box,
            text="1. Use the large 'Record from Mic' button above (Tejas)\n"
                 "2. Or click 'Upload .wav' to load a mixed track",
            style="Muted.TLabel",
            wraplength=280,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 6))

        # --- Microphone device selection (Tejas) ---
        if sd is None:
            ttk.Label(
                box,
                text="Microphone recording requires 'sounddevice'.",
                style="Muted.TLabel",
                wraplength=260,
            ).pack(anchor="w", padx=10, pady=(4, 4))
            self.device_box = None
            self.record_duration_var = None
            return

        ttk.Label(box, text="Mic device:", style="Muted.TLabel").pack(
            anchor="w", padx=10, pady=(6, 2)
        )

        devices = sd.query_devices()
        input_devices = [
            f"{i}: {d['name']}"
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(box, values=input_devices, textvariable=self.device_var, state="readonly")
        if input_devices:
            self.device_box.current(0)
        self.device_box.pack(fill="x", padx=10, pady=(0, 4))

        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(row, text="Duration (s):", style="Muted.TLabel").pack(side="left")
        self.record_duration_var = tk.StringVar(value="5")
        ttk.Entry(row, textvariable=self.record_duration_var, width=6).pack(side="left", padx=(4, 0))

        ttk.Label(
            box,
            text="After recording or uploading, the app will:\n"
                 "• run Aidan's separation model\n"
                 "• compute Nico's features\n"
                 "• generate Perlin fields for each tab",
            style="Muted.TLabel",
            wraplength=280,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(4, 10))

    def _mode_section(self, parent):
        box = ttk.Labelframe(parent, text="Visualization Mode", style="Card.TLabelframe")
        box.pack(fill="x", pady=8)
        ttk.Label(
            box,
            text="Currently only the Perlin field is implemented.\n"
                 "Other modes (waveform, spectrogram) could be added later.",
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor="w", padx=10, pady=(6, 8))

    # ---------- UI event handlers -------------------------------------

    def _on_upload(self):
        """
        User chooses a .wav that already contains violin+viola.
        """
        path = filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self._run_full_analysis(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze file:\n{e}")

    def _on_record(self):
        """
        User wants to record directly from a microphone (Tejas).
        """
        if sd is None:
            messagebox.showerror("Recording not available", "sounddevice is not installed.")
            return

        # Parse duration
        try:
            duration = float(self.record_duration_var.get())
            duration = max(1.0, min(60.0, duration))  # 1–60 seconds
        except Exception:
            messagebox.showerror("Invalid duration", "Please enter a number between 1 and 60.")
            return

        # Determine selected device index
        device_index = None
        if hasattr(self, "device_box") and self.device_box is not None and self.device_var.get():
            try:
                device_index = int(self.device_var.get().split(":", 1)[0])
            except Exception:
                device_index = None

        try:
            signal_arr, fs = record_audio_tejas(duration=duration, sample_rate=44100, channels=1, device=device_index)
        except Exception as e:
            messagebox.showerror("Recording error", f"Could not record audio:\n{e}")
            return

        # Save to a temporary wav file
        out_path = os.path.abspath("timbermind_recorded.wav")
        save_recording_to_wav(signal_arr, fs, out_path)
        messagebox.showinfo("Recording saved", f"Saved recording as:\n{out_path}")

        # Run the full separation + visualization pipeline
        try:
            self._run_full_analysis(out_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze recording:\n{e}")

    def _run_full_analysis(self, mix_path):
        """
        This is the main glue that connects:
        Tejas  -> Aidan -> Nico
        --------------------------------
        1) Take the mixed WAV file (recorded or uploaded)
        2) Run Aidan's NMF separation (2 stems)
        3) Load all three signals into memory
        4) Let Nico compute features for each
        5) Tell visualizer + audio controller to rebuild
        """
        if sf is None:
            raise RuntimeError("soundfile (pysoundfile) is required for analysis.")

        # --- 1. Aidan: run source separation on the mixed file
        base, _ = os.path.splitext(mix_path)
        out_prefix = base + "_tm"
        stem_paths, sr = separate_file(mix_path, out_prefix)
        if len(stem_paths) < 2:
            raise RuntimeError("Source separation did not produce two stems.")

        # Interpret stems:
        violin_path = stem_paths[0]  # labeled source1
        viola_path = stem_paths[1]   # labeled source2

        # --- 2. Load the three WAVs as mono float arrays
        mix_signal, sr_mix = load_audio_mono(mix_path)
        violin_signal, sr_vn = load_audio_mono(violin_path)
        viola_signal, sr_va = load_audio_mono(viola_path)
        assert sr_mix == sr_vn == sr_va, "Sample rates of stems must match."

        # --- 3. Nico: compute features for each view
        self.feature_bank.compute_for_signal("mix", mix_signal, sr_mix)
        self.feature_bank.compute_for_signal("violin", violin_signal, sr_mix)
        self.feature_bank.compute_for_signal("viola", viola_signal, sr_mix)

        # --- 4. Update audio controller
        self.audio_ctrl.set_signals(mix_signal, violin_signal, viola_signal, sr_mix)

        # --- 5. Ask visualizer to rebuild fields
        self.bus.dispatch("app:newAudio", None)

        messagebox.showinfo(
            "Analysis complete",
            "Source separation and feature extraction are done.\n\n"
            "Use Play / Pause and switch tabs to explore the audio and fields."
        )

    def _on_play(self):
        self.audio_ctrl.play()
        self.bus.dispatch("app:play", None)

    def _on_pause(self):
        self.audio_ctrl.pause()
        self.bus.dispatch("app:pause", None)
        self._refresh_time()

    def _on_reset(self):
        """
        Reset the canvases so a new clip can be drawn from scratch.
        """
        self.visualizer.clear_canvases()
        self.audio_ctrl.current_time = 0.0
        self.state.current = 0.0
        self._refresh_time()

    def _on_tab_change(self, _event=None):
        """
        When the user clicks a different tab, tell everyone which
        logical view is now active. AudioController will switch
        audible playback to that view if we are currently playing.
        """
        idx = self.notebook.index(self.notebook.select())
        view_key = {0: "mix", 1: "violin", 2: "viola"}.get(idx, "mix")
        self.state.active_view = view_key
        self.bus.dispatch("app:setActiveView", {"view": view_key})

    def _on_media_state(self, detail):
        if not detail:
            return
        self.state.duration = detail.get("duration", self.state.duration)
        self.state.current = detail.get("currentTime", self.state.current)
        self._refresh_time()

    def _refresh_time(self):
        self.time_lbl.config(
            text=f"{mmss(self.state.current)} / {mmss(self.state.duration)}"
        )


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    bus = EventBus()
    state = AppState()
    feature_bank = FeatureBank()
    audio_ctrl = AudioController(bus, state)

    # Temporary canvases; real ones will be attached by the UI
    dummy = tk.Tk()
    dummy.withdraw()
    dummy_canvas = tk.Canvas(dummy)
    visualizer = PerlinVisualizer(
        bus,
        state,
        feature_bank,
        dummy_canvas,
        dummy_canvas,
        dummy_canvas,
    )
    dummy.destroy()

    app = VisualizerApp(bus, state, feature_bank, audio_ctrl, visualizer)
    playback = PlaybackLoop(bus, state, audio_ctrl, visualizer)

    app.mainloop()
