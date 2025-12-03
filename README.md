# TimberMind – Instrument Visualizer (V1)

TimberMind is a desktop app that **records or loads audio**, **separates it into two stems**, and **visualizes the timbre over time** with a Perlin-noise flow field. It’s designed for explaining what’s happening inside a simple source-separation pipeline in a way that’s visually intuitive.

> **Team roles**
>
> - **Tejas** – Microphone recording & WAV handling  
> - **Aidan** – NMF-based source separation (mixed audio → 2 stems)  
> - **Nico** – Audio feature extraction & Perlin-noise visualizer  
> - **Jan** – Tkinter UI & event wiring  

---

## ✨ Features

- 🎤 **Record from microphone**  
  - Large “Record from Mic” button in the top bar  
  - Device selection & configurable duration  

- 🎧 **Load existing WAV files**  
  - Upload a mixed violin+viola recording (`.wav`)  
  - Automatically analyzed by the pipeline  

- 🔀 **Source separation (NMF)**  
  - Uses `sklearn.decomposition.NMF` on the magnitude STFT  
  - Produces **two stem WAVs**:
    - `*_source1.wav` → labeled as **“Violin”**
    - `*_source2.wav` → labeled as **“Viola”**  
  - Labels are for visualization; mathematically they’re just two NMF components  

- 📊 **Feature extraction** (per view)  
  For each of: **mix**, **violin**, and **viola**:
  - RMS loudness (frame-wise)
  - Spectral centroid (brightness)
  - Normalized so visuals scale nicely across clips

- 🎨 **Perlin-noise visualizer**  
  - One Perlin field per tab:
    - **Original mix**
    - **Viola**
    - **Violin**
  - Features control:
    - RMS → line thickness (louder = thicker strokes)  
    - Centroid → stroke length & color blend  
    - Time → horizontal reveal (left to right across the canvas)
  - Field is precomputed once per clip for smooth playback

- 🖥️ **Tkinter UI**  
  - Dark, minimal layout  
  - Top bar: **Record, Upload, Reset, Play/Pause, time display**  
  - 3-tab notebook for **Original mix / Viola / Violin** canvases  
  - Uses a simple **EventBus** so UI, audio, and visuals are loosely coupled

---

## 🧱 Tech Stack

- **Language:** Python 3.10+
- **GUI:** Tkinter (`tk`, `ttk`)
- **Audio I/O:** `sounddevice`, `soundfile`
- **DSP:** `numpy`, `scipy`
- **ML / Separation:** `scikit-learn` (`NMF`)
- **Visualization:** Tkinter `Canvas` + custom Perlin noise implementation

---

## 📁 Project Structure (example)

You can adjust this to your actual repo layout, but a typical structure looks like:

```text
TimberMind/
├─ main.py                # <- the big integrated app shown in the prompt
├─ requirements.txt
├─ README.md
├─ data/                  # (optional) example .wav files
└─ docs/                  # (optional) screenshots, diagrams, report PDFs
If your main file is not main.py, just update the run command in the sections below.

⚙️ Installation
1. Clone the repository
bash
Copy code
git clone https://github.com/<your-username>/TimberMind.git
cd TimberMind
2. Create a virtual environment
Windows (PowerShell / CMD):

bash
Copy code
python -m venv .venv
.venv\Scripts\activate
macOS / Linux:

bash
Copy code
python3 -m venv .venv
source .venv/bin/activate
You should see (.venv) at the start of your terminal prompt when it’s active.

3. Install dependencies
Make sure you have a requirements.txt with:

txt
Copy code
numpy
scipy
sounddevice
soundfile
matplotlib
scikit-learn
Then install:

bash
Copy code
pip install -r requirements.txt
tkinter is not in requirements.txt because it comes from your Python/OS install.

On Ubuntu/Debian you may need: sudo apt install python3-tk

▶️ Running the App
From the project root (with the virtualenv activated):

bash
Copy code
python main.py
(or whatever your main file is called, e.g. python timbermind_app.py)

If everything is set up correctly, a window titled “TimberMind – Instrument Visualizer” will open.

🧩 How It Works (High-Level)
Choose input

Click “Record from Mic” to record a new clip

Or click “Upload .wav” to load an existing recording

Source separation (Aidan)

Mixed audio is converted to a magnitude spectrogram via STFT (scipy.signal.stft)

NMF (sklearn.decomposition.NMF) factorizes it into 2 components

Components are turned back into time-domain signals with soft masking + inverse STFT

Two stem files are written:

<base>_tm_source1.wav → visualized as “Violin”

<base>_tm_source2.wav → visualized as “Viola”

Feature extraction (Nico)

For mix, violin, and viola:

Slice audio into overlapping windows

Compute RMS and magnitude spectrum

Compute spectral centroid from the magnitude spectrum

Store times, rms, and centroid per frame in a FeatureBank

Perlin field generation

Each tab has a Perlin-based flow field built offline:

A set of particles flow across the canvas

Direction is given by Perlin noise

At each step, the current time → look up features in FeatureBank

Features control:

Stroke thickness (RMS)

Stroke length (centroid)

Stroke color (centroid blend between a low-color and high-color)

Playback & visualization

AudioController manages audio arrays & global playback time

PlaybackLoop steps the clock and calls PerlinVisualizer.render_frame()

The active tab’s canvas reveals more strokes from left to right as time progresses

Switching tabs:

Keeps the same playback time

Changes which stem you hear and which canvas is drawn onto

🕹️ Usage Tips
Recording

Choose a device in the left “Input / Analysis” panel

Set Duration (s)

Click “Record from Mic”

After it finishes, the app will automatically separate and visualize the clip

Uploading

Click “Upload .wav” and pick a mixed recording

A message box will confirm when analysis is complete

Tabs

Original mix – whole mix visualized

Viola – first stem (labeled Viola in UI)

Violin – second stem (labeled Violin in UI)

Reset Canvas

Clears all three canvases so you can re-draw the fields from scratch

Does not delete audio; you can hit Play again

🧪 Known Limitations / Future Work
NMF separation is unsupervised; stems are not guaranteed to be “true” violin/viola, they’re just two learned components.

No model training is done inside this app; it operates purely on the loaded clip.

Currently only Perlin visualization mode is implemented; waveform / spectrogram views are possible future additions.

No persistent project saving yet – all processing happens in-memory per session.

👥 Contributors
Tejas – Microphone recording & audio input

Aidan – NMF-based source separation

Nico – Feature extraction & Perlin field visualization

Jan – Tkinter UI, layout, and event wiring
