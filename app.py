import os
import time
import math
import random
import importlib.util
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser

# We check for optional audio libraries without try/except so the file stays easy to read.
# If soundfile is not installed, we simply set it to None and the app will use a safe fallback.
sf = None
if importlib.util.find_spec("soundfile"):
    import soundfile as sf  # type: ignore


# --- Helper function to format time
def mmss(sec):
    sec = int(max(0, sec or 0))
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"


# --- Simple event system to connect UI with teammates later
class EventBus:
    def __init__(self):
        self._subs = {}

    def subscribe(self, name, fn):
        self._subs.setdefault(name, []).append(fn)

    def dispatch(self, name, detail=None):
        for fn in self._subs.get(name, []):
            try:
                fn(detail)
            except Exception as e:  # pragma: no cover
                print(f"[EVENT ERROR] {name}: {e}")


# --- Shared state for UI
class AppState:
    def __init__(self):
        self.page = "home"
        self.mode = "perlin"
        self.status = "stopped"
        self.duration = 0
        self.current = 0
        self.library = []  # list of file paths


# --- Audio analysis helpers

def compute_spectrogram(signal, sample_rate, win_size=1024, hop=512):
    """Turn a 1-D audio clip into a magnitude spectrogram by hand."""
    # If we have no audio, we return a quiet placeholder so the app stays stable.
    if signal is None or len(signal) == 0:
        return np.zeros((win_size // 2 + 1, 1)), np.array([0.0]), np.array([0.0])
    # Work out how many time windows fit inside the audio.
    frames = 1 + max(0, (len(signal) - win_size) // hop)
    # A Hann window keeps the edges of each slice smooth.
    window = np.hanning(win_size)
    # We build an empty spectrogram that we will fill column by column.
    spec = np.zeros((win_size // 2 + 1, frames), dtype=float)
    for i in range(frames):
        start = i * hop
        frame = signal[start: start + win_size]
        if len(frame) < win_size:
            frame = np.pad(frame, (0, win_size - len(frame)))
        # Apply the smooth window so we avoid sharp clicks between frames.
        frame = frame * window
        spectrum = np.fft.rfft(frame)
        # We only need the magnitude for this visual task.
        spec[:, i] = np.abs(spectrum)
    freqs = np.fft.rfftfreq(win_size, 1 / sample_rate)
    times = np.arange(frames) * hop / float(sample_rate)
    return spec, freqs, times


def separate_instruments(mag_spec: np.ndarray, sample_rate: int) -> dict:
    """
    mag_spec: 2D numpy array, shape (F, T) = frequency bins × time frames
    returns: {
        "violin": mask_violin,  # shape (F, T), values 0–1
        "viola":  mask_viola,   # shape (F, T), values 0–1
    }
    """
    # If there is nothing to analyze we hand back empty masks.
    if mag_spec.size == 0:
        empty = np.zeros_like(mag_spec)
        return {"violin": empty, "viola": empty}

    # We fake the model by drawing gentle random patterns so visuals still move.
    rng = np.random.default_rng(42)
    noise_vln = rng.random(mag_spec.shape)
    noise_vla = rng.random(mag_spec.shape)

    # Bias the top of the spectrum toward violin and the bottom toward viola
    # so the shapes look slightly different across the canvas.
    bias = np.linspace(0.6, 1.4, mag_spec.shape[0])[:, None]
    mask_violin = noise_vln * bias
    mask_viola = noise_vla * (2 - bias)

    # Normalize so both masks add up to 1.0 at every time and frequency bin.
    total = mask_violin + mask_viola + 1e-8
    mask_violin /= total
    mask_viola /= total
    return {"violin": mask_violin, "viola": mask_viola}


class FeatureController:
    def __init__(self, bus, state, audio_ctrl):
        # We keep references to the bus, shared state, and audio helper so we can react to events.
        self.bus = bus
        self.state = state
        self.audio_ctrl = audio_ctrl
        self.features = None
        self.max_energy = {"violin": 1.0, "viola": 1.0}
        self.max_centroid = 1.0
        # Listen for file loads and play events so we know when to refresh features.
        bus.subscribe("app:loadFile", self._on_update_request)
        bus.subscribe("audio:updated", self._on_update_request)
        bus.subscribe("app:play", self._on_update_request)

    def _on_update_request(self, _detail):
        # Any new audio or play command triggers a fresh analysis.
        self.analyze()

    def analyze(self):
        # Pull the latest mono signal and sample rate from the audio controller.
        signal, sr = self.audio_ctrl.get_signal()
        if signal is None or sr is None:
            self.features = None
            return
        # Build a spectrogram, then ask our stub model for instrument masks.
        mag_spec, freqs, times = compute_spectrogram(signal, sr)
        masks = separate_instruments(mag_spec, sr)
        # Apply each mask to carve out instrument-specific magnitude spectrograms.
        inst_specs = {
            name: masks.get(name, np.zeros_like(mag_spec)) * mag_spec
            for name in ["violin", "viola"]
        }

        feats = {"times": times}
        for inst, spec in inst_specs.items():
            # Simple frame-wise features: energy, centroid, and mask activity.
            energy = np.sum(spec ** 2, axis=0)
            centroid_num = np.sum(spec * freqs[:, None], axis=0)
            centroid_den = np.sum(spec + 1e-9, axis=0)
            centroid = centroid_num / centroid_den
            activity = np.mean(masks.get(inst, np.zeros_like(spec)), axis=0)
            feats[inst] = {
                "energy": energy,
                "centroid": centroid,
                "activity": activity,
            }
            # Track peaks so we can normalize visuals later.
            self.max_energy[inst] = max(float(np.max(energy)), 1e-6)
        self.max_centroid = max(float(np.max(freqs)), 1e-6)
        self.features = feats

    def get_features_at_time(self, t: float) -> dict:
        # If analysis is not ready, return safe defaults.
        if not self.features or "times" not in self.features:
            return {
                "times": np.array([0.0]),
                "violin": {"energy": 0.0, "centroid": 0.0, "activity": 0.0},
                "viola": {"energy": 0.0, "centroid": 0.0, "activity": 0.0},
            }
        times = self.features["times"]
        if len(times) == 0:
            return {
                "times": np.array([0.0]),
                "violin": {"energy": 0.0, "centroid": 0.0, "activity": 0.0},
                "viola": {"energy": 0.0, "centroid": 0.0, "activity": 0.0},
            }
        t = float(t)
        result = {"times": times}
        for inst in ["violin", "viola"]:
            inst_data = self.features.get(inst, {})
            # Interpolate so visuals stay smooth between frames.
            energy = np.interp(t, times, inst_data.get("energy", np.zeros_like(times)))
            centroid = np.interp(t, times, inst_data.get("centroid", np.zeros_like(times)))
            activity = np.interp(t, times, inst_data.get("activity", np.zeros_like(times)))
            result[inst] = {
                "energy": energy,
                "centroid": centroid,
                "activity": activity,
            }
        return result


class AudioController:
    def __init__(self, bus, state):
        # Store the shared bus and state so we can keep UI and audio in sync.
        self.bus = bus
        self.state = state
        self.signal = None
        self.sample_rate = None
        self.duration = 0
        self.current_time = 0.0
        self.playing = False
        # Listen for file loading and transport controls.
        bus.subscribe("app:loadFile", self._on_load)
        bus.subscribe("app:play", self._on_play)
        bus.subscribe("app:pause", self._on_pause)

    def _on_load(self, detail):
        # When a user picks a file, read it into a mono numpy array.
        path = detail.get("file_path") if detail else None
        if not path:
            return
        self.signal, self.sample_rate = self._read_audio(path)
        if self.signal is not None and self.sample_rate:
            self.duration = len(self.signal) / float(self.sample_rate)
            self.current_time = 0.0
            self.bus.dispatch("audio:updated", None)
            self.dispatch_state()

    def _on_play(self, _detail):
        # On play, make sure we have at least a short silent clip.
        if self.signal is None:
            self.signal = np.zeros(44100)
            self.sample_rate = 44100
            self.duration = 1.0
            self.bus.dispatch("audio:updated", None)
        self.playing = True
        self.dispatch_state()

    def _on_pause(self, _detail):
        # Pause simply freezes the timer and updates the UI.
        self.playing = False
        self.dispatch_state()

    def step(self, dt):
        # Advance the playback clock by dt seconds when playing.
        if not self.playing:
            return self.current_time
        self.current_time += dt
        if self.duration > 0 and self.current_time > self.duration:
            self.current_time = self.current_time % self.duration
        return self.current_time

    def dispatch_state(self):
        # Let the UI know our play state, duration, and current time.
        self.bus.dispatch(
            "media:state",
            {
                "status": "playing" if self.playing else "paused",
                "duration": self.duration,
                "currentTime": self.current_time,
            },
        )

    def get_signal(self):
        return self.signal, self.sample_rate

    def _read_audio(self, path):
        # Try to load audio with soundfile first, then fall back to the built-in wave reader.
        ext = os.path.splitext(path)[1].lower()
        if sf is not None:
            try:
                data, sr = sf.read(path)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                return data.astype(float), int(sr)
            except Exception:
                pass
        if ext == ".wav":
            try:
                import wave

                with wave.open(path, "rb") as w:
                    sr = w.getframerate()
                    frames = w.readframes(w.getnframes())
                    data = np.frombuffer(frames, dtype=np.int16)
                    if w.getnchannels() > 1:
                        data = data.reshape(-1, w.getnchannels())
                        data = np.mean(data, axis=1)
                    data = data.astype(float) / 32768.0
                    return data, sr
            except Exception:
                pass
        # As a last resort, hand back gentle noise so the visuals still run.
        rng = np.random.default_rng(0)
        return rng.normal(0, 0.01, 44100).astype(float), 44100


class PerlinVisualizer:
    def __init__(self, bus, state, feature_ctrl, canvas_combined: tk.Canvas, canvas_violin: tk.Canvas, canvas_viola: tk.Canvas):
        # Keep handles to the bus, shared state, and features so we can react to events.
        self.bus = bus
        self.state = state
        self.feature_ctrl = feature_ctrl
        self.canvases = {
            "combined": canvas_combined,
            "violin": canvas_violin,
            "viola": canvas_viola,
        }
        # Keep a reference for playback loop scheduling compatibility.
        self.canvas = canvas_combined
        self.config = {
            "colors": {"violin": "#a855f7", "viola": "#22d3ee"},
            "shape_style": "lines",
            "enabled": {"violin": True, "viola": True, "cello": False},
        }

        # Perlin field parameters tuned for slow, coherent flow.
        self.grid_size = 96
        self.noise_scale_x = 220.0
        self.noise_scale_y = 220.0
        self.flow_speed = 0.05
        self.gradients = self._generate_gradients()

        self.segments = {"combined": [], "violin": [], "viola": []}
        self.segment_indices = {"combined": 0, "violin": 0, "viola": 0}
        self.drawn_until = {"combined": 0, "violin": 0, "viola": 0}
        self._placeholder_mode = None
        self._build_job = None

        bus.subscribe("visual:updateConfig", self._on_config)
        bus.subscribe("app:setMode", self._on_mode)
        bus.subscribe("app:loadFile", self._on_new_audio)
        bus.subscribe("audio:updated", self._on_new_audio)
        bus.subscribe("app:play", self._on_play)

    # --- Event hooks
    def _on_play(self, _detail=None):
        # If playback restarts from the top, reveal from scratch without rebuilding segments.
        if (self.state.current or 0) <= 0.05:
            self._reset_drawn_only()

    def _on_new_audio(self, _detail=None):
        self.reset(wipe_segments=True)
        self._schedule_build()

    def _on_config(self, detail):
        if not detail:
            return
        if "instrument" in detail and "color" in detail:
            self.config["colors"][detail["instrument"]] = detail["color"]
        if "shape_style" in detail:
            self.config["shape_style"] = detail["shape_style"]
        if "enabled" in detail:
            for inst, flag in detail["enabled"].items():
                self.config["enabled"][inst] = flag

    def _on_mode(self, detail):
        if detail and detail.get("mode"):
            self.state.mode = detail["mode"]
            if self.state.mode == "perlin":
                self._placeholder_mode = None
                self._reset_drawn_only()
            else:
                self._show_placeholder(self.state.mode)

    # --- Reset and scheduling
    def reset(self, wipe_segments=True):
        for canvas in self.canvases.values():
            canvas.delete("all")
        self.drawn_until = {k: 0 for k in self.drawn_until}
        self.segment_indices = {k: 0 for k in self.segment_indices}
        if wipe_segments:
            self.segments = {"combined": [], "violin": [], "viola": []}
        self._placeholder_mode = None

    def _reset_drawn_only(self):
        for canvas in self.canvases.values():
            canvas.delete("all")
        self.drawn_until = {k: 0 for k in self.drawn_until}
        self.segment_indices = {k: 0 for k in self.segment_indices}

    def _schedule_build(self):
        if self._build_job:
            self.canvas.after_cancel(self._build_job)
        self._build_job = self.canvas.after(50, self._build_segments)

    # --- Perlin helpers
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

    # --- Offline stroke construction
    def _build_segments(self):
        self._build_job = None
        width = max(400, self.canvases["combined"].winfo_width() or 1200)
        height = max(300, self.canvases["combined"].winfo_height() or 700)
        duration = max(self.state.duration, 1.0)
        self.gradients = self._generate_gradients()

        violin_color = self.config["colors"].get("violin", "#a855f7")
        viola_color = self.config["colors"].get("viola", "#22d3ee")

        # Generate both instruments across the full canvas width.
        violin_segments = self._generate_segments_for_instrument(
            "violin", violin_color, 0, width, width, height, duration
        )
        viola_segments = self._generate_segments_for_instrument(
            "viola", viola_color, 0, width, width, height, duration
        )

        # Store per-instrument segments for their dedicated canvases.
        self.segments["violin"] = violin_segments
        self.segments["viola"] = viola_segments

        # Combined view overlays both sets of segments in the same coordinate space.
        self.segments["combined"] = viola_segments + violin_segments

        # Sort for efficient reveal by x position.
        for key in self.segments:
            self.segments[key].sort(key=lambda seg: seg[0])
        self._reset_drawn_only()

    def _generate_segments_for_instrument(self, inst, color, x_min, x_max, width, height, duration):
        rng = random.Random(123 if inst == "violin" else 321)
        particles = 420
        steps = 160
        segments = []
        for _ in range(particles):
            x = x_min + rng.random() * (x_max - x_min)
            y = rng.random() * height
            for _ in range(steps):
                t = max(0.0, min(duration, (x / float(width)) * duration))
                feats = self.feature_ctrl.get_features_at_time(t)
                inst_feat = feats.get(inst, {})
                energy = float(inst_feat.get("energy", 0.0))
                centroid = float(inst_feat.get("centroid", 0.0))
                activity = float(inst_feat.get("activity", 0.0))
                energy_norm = min(1.0, energy / (self.feature_ctrl.max_energy.get(inst, 1.0)))
                centroid_norm = min(1.0, centroid / (self.feature_ctrl.max_centroid or 1.0))

                nx = (x / self.noise_scale_x) + t * self.flow_speed
                ny = (y / self.noise_scale_y) + t * self.flow_speed
                angle = self._perlin(nx % self.grid_size, ny % self.grid_size) * math.tau
                base_step = 1.2 + energy_norm * 2.2 + centroid_norm * 0.6
                if self.config.get("shape_style") == "particles":
                    base_step *= 0.6
                step = base_step
                x1 = x + math.cos(angle) * step
                y1 = y + math.sin(angle) * step
                y1 = max(0.0, min(height, y1))

                if x1 < x_min - 2 or x1 > x_max + 2:
                    x = x_min + rng.random() * (x_max - x_min)
                    y = rng.random() * height
                    continue

                width_px = 0.8 + activity * 2.0
                if self.config.get("shape_style") == "abstract":
                    width_px *= 1.1 + rng.random() * 0.3
                seg_max = max(x, x1)
                segments.append((seg_max, x, y, x1, y1, color, width_px, inst))
                x, y = x1, y1
        return segments

    # --- Rendering
    def _show_placeholder(self, mode_label: str):
        if self._placeholder_mode == mode_label:
            return
        self._placeholder_mode = mode_label
        for canvas in self.canvases.values():
            canvas.delete("all")
            canvas.create_text(
                canvas.winfo_width() // 2,
                canvas.winfo_height() // 2,
                text=f"{mode_label.title()} view coming soon",
                fill="#6b7280",
                font=("Segoe UI", 12, "bold"),
            )

    def render_frame(self, current_time: float):
        if self.state.mode != "perlin":
            self._show_placeholder(self.state.mode)
            return
        if not any(self.segments.values()):
            # Build on first draw if needed.
            if self._build_job is None:
                self._schedule_build()
            return

        duration = max(self.state.duration, 1e-6)
        progress = max(0.0, min(1.0, current_time / duration))

        for view, canvas in self.canvases.items():
            width = max(1, canvas.winfo_width())
            x_limit = progress * width
            idx = self.segment_indices.get(view, 0)
            segments = self.segments.get(view, [])
            while idx < len(segments) and segments[idx][0] <= x_limit:
                _, x0, y0, x1, y1, color, w, inst = segments[idx]
                if view != "combined" and inst != view:
                    idx += 1
                    continue
                if view == "combined" and not self.config["enabled"].get(inst, False):
                    idx += 1
                    continue
                if view != "combined" and not self.config["enabled"].get(inst, False):
                    idx += 1
                    continue

                if self.config.get("shape_style") == "particles":
                    size = max(1.0, w)
                    canvas.create_oval(x1 - size, y1 - size, x1 + size, y1 + size, fill=color, outline="")
                else:
                    canvas.create_line(x0, y0, x1, y1, fill=color, width=w, smooth=True, capstyle=tk.ROUND)
                idx += 1
            self.segment_indices[view] = idx
            self.drawn_until[view] = x_limit


class PlaybackLoop:
    def __init__(self, bus, state, audio_ctrl, visualizer):
        # Connect controllers so we can tick audio time and redraw visuals.
        self.bus = bus
        self.state = state
        self.audio_ctrl = audio_ctrl
        self.visualizer = visualizer
        self.running = False
        self.last_time = None
        # Start or stop the loop when the transport buttons are used.
        bus.subscribe("app:play", self._start)
        bus.subscribe("app:pause", self._stop)

    def _start(self, _detail=None):
        if self.running:
            return
        # Remember when the loop started so we can compute frame-to-frame dt.
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
        # Advance audio time, draw the visuals, and schedule the next frame.
        current_time = self.audio_ctrl.step(dt)
        self.visualizer.render_frame(current_time)
        self.audio_ctrl.dispatch_state()
        self.visualizer.canvas.after(16, self._tick)


# --- Modernized single-page application window
class VisualizerApp(tk.Tk):
    def __init__(self, bus, state):
        super().__init__()
        self.bus = bus
        self.state = state
        self.title("Instrument Visualizer")
        self.geometry("1100x700")
        self.minsize(900, 560)

        # Core colors for a dark, minimal look.
        self.c_bg = "#0c0f16"
        self.c_panel = "#111827"
        self.c_line = "#1f2937"
        self.c_accent = "#4f46e5"
        self.c_text = "#e5e7eb"
        self.c_muted = "#9ca3af"

        # User-configurable bits we expose through the controls panel.
        self.mode_var = tk.StringVar(value="perlin")
        self.shape_style = tk.StringVar(value="lines")
        self.enabled_instruments = {
            "violin": tk.BooleanVar(value=True),
            "viola": tk.BooleanVar(value=True),
        }

        self._make_style()
        self._build_layout()

        # Keep the time label in sync with playback updates from the backend.
        self.bus.subscribe("media:state", self._on_media_state)
        self._refresh_time()
        # Share initial config with the visualization layer so defaults match UI.
        self._dispatch_enabled()
        self.bus.dispatch("visual:updateConfig", {"shape_style": self.shape_style.get()})
        self.bus.dispatch("app:setMode", {"mode": self.mode_var.get()})

    def _make_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=self.c_bg)
        style.configure("TLabel", background=self.c_bg, foreground=self.c_text, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=self.c_bg, foreground=self.c_muted, font=("Segoe UI", 10))
        style.configure("TButton", background=self.c_panel, foreground=self.c_text, borderwidth=0, padding=8)
        style.map("TButton", background=[("active", self.c_line)])
        style.configure("Card.TFrame", background=self.c_panel)
        style.configure("Card.TLabelframe", background=self.c_panel, foreground=self.c_text)
        style.configure("Card.TLabelframe.Label", background=self.c_panel, foreground=self.c_muted)
        style.configure("Time.TLabel", background=self.c_bg, foreground=self.c_muted, font=("Segoe UI", 10, "bold"))

    def _build_layout(self):
        # Top bar with title, transport controls, and timer.
        top = ttk.Frame(self, padding=(16, 12))
        top.pack(side="top", fill="x")
        ttk.Label(top, text="Instrument Visualizer", font=("Segoe UI", 14, "bold")).pack(side="left")
        ttk.Button(top, text="Upload", command=self._on_upload).pack(side="right", padx=(6, 0))
        ttk.Button(top, text="Pause", command=self._on_pause).pack(side="right", padx=6)
        ttk.Button(top, text="Play", command=self._on_play).pack(side="right", padx=6)
        self.time_lbl = ttk.Label(top, text="00:00 / 00:00", style="Time.TLabel")
        self.time_lbl.pack(side="right", padx=12)

        # Main area split into controls (left) and canvas (right).
        body = ttk.Frame(self, padding=(16, 0, 16, 16))
        body.pack(fill="both", expand=True)

        controls = ttk.Frame(body, width=300)
        controls.pack(side="left", fill="y", padx=(0, 16))

        self._input_section(controls)
        self._mode_section(controls)
        self._instrument_section(controls)
        self._style_section(controls)
        self._recent_section(controls)

        canvas_frame = ttk.Frame(body, style="Card.TFrame")
        canvas_frame.pack(side="right", fill="both", expand=True)
        canvas_frame.pack_propagate(False)
        notebook = ttk.Notebook(canvas_frame)
        notebook.pack(fill="both", expand=True, padx=4, pady=4)

        self.canvas_combined = tk.Canvas(notebook, bg="#0b0b0f", highlightthickness=1, highlightbackground=self.c_line)
        self.canvas_violin = tk.Canvas(notebook, bg="#0b0b0f", highlightthickness=1, highlightbackground=self.c_line)
        self.canvas_viola = tk.Canvas(notebook, bg="#0b0b0f", highlightthickness=1, highlightbackground=self.c_line)

        notebook.add(self.canvas_combined, text="Combined")
        notebook.add(self.canvas_violin, text="Violin only")
        notebook.add(self.canvas_viola, text="Viola only")

    def _input_section(self, parent):
        box = ttk.Labelframe(parent, text="Input", style="Card.TLabelframe")
        box.pack(fill="x", pady=8)
        ttk.Label(box, text="Recording only — upload a file to begin.", style="Muted.TLabel").pack(
            anchor="w", padx=10, pady=6
        )

    def _mode_section(self, parent):
        box = ttk.Labelframe(parent, text="Visualization Mode", style="Card.TLabelframe")
        box.pack(fill="x", pady=8)
        modes = ["perlin", "waveform", "spectrogram", "lissajous"]
        ttk.Label(box, text="Pick how to draw the audio:", style="Muted.TLabel").pack(anchor="w", padx=10, pady=(4, 2))
        mode_cb = ttk.Combobox(box, values=modes, textvariable=self.mode_var, state="readonly")
        mode_cb.pack(fill="x", padx=10, pady=(0, 8))
        mode_cb.bind("<<ComboboxSelected>>", self._on_mode_change)

    def _instrument_section(self, parent):
        box = ttk.Labelframe(parent, text="Instruments", style="Card.TLabelframe")
        box.pack(fill="x", pady=8)
        ttk.Label(box, text="Toggle instruments in the scene:", style="Muted.TLabel").pack(anchor="w", padx=10, pady=(4, 2))
        for name, var in self.enabled_instruments.items():
            ttk.Checkbutton(box, text=name.capitalize(), variable=var, command=self._on_instrument_toggle).pack(
                anchor="w", padx=12, pady=2
            )

    def _style_section(self, parent):
        box = ttk.Labelframe(parent, text="Colors and Style", style="Card.TLabelframe")
        box.pack(fill="x", pady=8)
        ttk.Button(box, text="Violin color...", command=lambda: self._pick_color("violin")).pack(fill="x", padx=10, pady=(6, 2))
        ttk.Button(box, text="Viola color...", command=lambda: self._pick_color("viola")).pack(fill="x", padx=10, pady=(2, 8))
        ttk.Label(box, text="Shape style", style="Muted.TLabel").pack(anchor="w", padx=10, pady=(0, 2))
        style_cb = ttk.Combobox(box, values=["lines", "particles", "abstract"], textvariable=self.shape_style, state="readonly")
        style_cb.pack(fill="x", padx=10, pady=(0, 10))
        style_cb.bind("<<ComboboxSelected>>", self._on_shape_style)

    def _recent_section(self, parent):
        box = ttk.Labelframe(parent, text="Recent files", style="Card.TLabelframe")
        box.pack(fill="both", expand=True, pady=8)
        self.recent_list = tk.Listbox(
            box,
            bg=self.c_panel,
            fg=self.c_text,
            highlightthickness=1,
            highlightbackground=self.c_line,
            selectbackground=self.c_accent,
        )
        self.recent_list.pack(fill="both", expand=True, padx=8, pady=8)

    def _on_upload(self):
        # Let the user choose an audio file and hand it to the audio controller.
        path = filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac"), ("All files", "*.*")],
        )
        if not path:
            return
        self.state.library.append(path)
        self.recent_list.insert(tk.END, os.path.basename(path))
        self.bus.dispatch("app:loadFile", {"file_path": path})

    def _on_play(self):
        self.bus.dispatch("app:play", None)
        self.state.status = "playing"
        self._refresh_time()

    def _on_pause(self):
        self.bus.dispatch("app:pause", None)
        self.state.status = "paused"
        self._refresh_time()

    def _on_mode_change(self, _event=None):
        mode = self.mode_var.get()
        self.state.mode = mode
        self.bus.dispatch("app:setMode", {"mode": mode})

    def _on_shape_style(self, _event=None):
        self.bus.dispatch("visual:updateConfig", {"shape_style": self.shape_style.get()})

    def _dispatch_enabled(self):
        enabled = {name: var.get() for name, var in self.enabled_instruments.items()}
        self.bus.dispatch("visual:updateConfig", {"enabled": enabled})

    def _on_instrument_toggle(self):
        self._dispatch_enabled()

    def _pick_color(self, instrument):
        color = colorchooser.askcolor(title=f"Choose {instrument} color")
        if color and color[1]:
            self.bus.dispatch("visual:updateConfig", {"instrument": instrument, "color": color[1]})

    def _on_media_state(self, detail):
        if not detail:
            return
        self.state.duration = detail.get("duration", self.state.duration)
        self.state.current = detail.get("currentTime", self.state.current)
        self._refresh_time()

    def _refresh_time(self):
        self.time_lbl.config(text=f"{mmss(self.state.current)} / {mmss(self.state.duration)}")


if __name__ == "__main__":
    bus = EventBus()
    state = AppState()
    # Start the fresh UI first so we can pass its canvas to the visualizer.
    app = VisualizerApp(bus, state)

    audio_ctrl = AudioController(bus, state)
    feature_ctrl = FeatureController(bus, state, audio_ctrl)
    visualizer = PerlinVisualizer(
        bus,
        state,
        feature_ctrl,
        app.canvas_combined,
        app.canvas_violin,
        app.canvas_viola,
    )
    playback = PlaybackLoop(bus, state, audio_ctrl, visualizer)

    app.mainloop()
