Instrument Visualizer — Install & Run Guide

This README explains how to set up and run the Instrument Visualizer prototype on your own computer. It covers:

Installing Python and dependencies

Setting up a virtual environment

Running each team member’s component (audio input, UI, Perlin visualizer / model output)

The instructions assume Windows, but the same steps work on macOS/Linux with minor command changes.

1. What’s in this project?

This prototype combines four main pieces:

Audio input module (Tejas)

Records from the microphone or loads a .wav file.

Outputs a 1D NumPy array (digital_signal) and a sample rate so other modules can treat audio as data.

User Interface (Jan Paul)

A Tkinter desktop app (instrument_ui.py) with:

Top bar (Upload / Play / Pause)

Sidebar pages: Home, Visualize, Models, Library, Settings

An internal event bus so teammates can hook in playback, ML, and visualization.

Instrument classification model (Aidan)

CNN trained on mel-spectrograms to classify cello, viola, violin 1, violin 2.

Exposes model outputs that can later be visualized (e.g., which instrument is playing when).

Perlin-noise visualizer (Nicholas)

Takes audio features (loudness, spectral centroid, etc.) and generates a Perlin flow field image or live animation that responds to the sound.

2. Prerequisites
2.1 Install Python

Go to the official Python site and download Python 3.10 or newer.

During installation on Windows, check:

✅ “Add Python to PATH”

To confirm it worked, open Command Prompt or PowerShell and run:

python --version


You should see something like Python 3.10.x or higher.

On macOS/Linux, use python3 instead of python.

3. Create a project folder

Create a folder where you’ll keep everything, for example:

InstrumentVisualizer/
    audio_input/         # Tejas’ recording/loading scripts
    ui/                  # Jan Paul's instrument_ui.py
    visualizer/          # Nicholas’ Perlin visualizer
    model/               # Aidan's CNN/model code (optional for now)


Place the provided files into the appropriate subfolders. At minimum you should have:

audio_input/tejas_audio.py (or similar, containing TEJAS SOURCE CODE V1 and V2)

ui/instrument_ui.py

(Names don’t have to match exactly, but keep them logical.)

4. Set up a virtual environment

From the root of the project (InstrumentVisualizer/):

On Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate

On macOS / Linux (terminal)
python3 -m venv .venv
source .venv/bin/activate


When activated, your prompt will show something like (.venv) at the beginning.

To deactivate later, just run:

deactivate

5. Install Python dependencies

With the virtual environment activated, install the required packages.

5.1 Audio input & basic visualization

Tejas’ code and the simple plot example need:

sounddevice — record from microphone

numpy — numerical arrays

scipy — wavfile.read() for .wav files

matplotlib — optional plotting

Install them:

pip install sounddevice numpy scipy matplotlib


On some systems you may also need to install system audio drivers or permissions for microphone access.

5.2 UI

instrument_ui.py only uses the Python standard library (tkinter, ttk, os, filedialog), so no extra pip packages are required. Tkinter comes with most Python installs.

If Tkinter is missing, install it via your OS package manager (e.g., sudo apt install python3-tk on Ubuntu).

5.3 Optional: extra libs for Perlin visualizer / model

Depending on how far you integrate Nicholas’ and Aidan’s code, you may also need (typical for that repo):

pip install soundfile pygame


If/when you bring in the CNN model, you’ll also install whatever ML library it uses (e.g., torch, torchaudio, etc.), but that’s optional for basic demo.

6. Running each component
6.1 Audio input (Tejas)

Inside audio_input/, you should have the functions:

record_audio(duration=5, sample_rate=44100, channels=1) — uses sounddevice to record live mic audio, returns digital_signal, sample_rate.

load_audio(file_path) — uses scipy.io.wavfile.read to load a .wav, normalize it, and return digital_signal, sample_rate.

A minimal test script (you can add this at the bottom of Tejas’ file):

if __name__ == "__main__":
    # Example: record 5 seconds from microphone
    digital_signal, sr = record_audio(duration=5)
    print("Recorded", len(digital_signal), "samples at", sr, "Hz")

    # Example: plot the waveform (optional)
    import matplotlib.pyplot as plt
    plt.plot(digital_signal)
    plt.title("Recorded audio")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.show()


Run from the project root with:

python -m audio_input.tejas_audio


(or python tejas_audio.py if you run it directly in that folder.)

How to use it (from the report)

Open terminal / PowerShell.

Navigate to the script folder.

Run the script.

Speak during the 5-second window.

The output will be stored as a digital array (digital_signal) that other modules can consume.

6.2 UI shell (Jan Paul)

Inside ui/, you have instrument_ui.py. It creates:

A dark theme app window with top bar, navigation, and multiple pages.

An EventBus so the UI can call external logic via events (app:loadFile, app:play, app:pause, app:setMode).

Run it from the project root:

python -m ui.instrument_ui


or from inside ui/:

python instrument_ui.py


What you should see:

A window titled “Instrument Visualizer — UI Shell”.

Top-left: circular IV logo and title “Instrument UI — Student Edition”.

Top-right buttons: Upload, Play, Pause, 00:00 / 00:00.

Left sidebar with navigation: Home, Visualize, Models, Library, Settings.

From here:

Click Upload → choose an audio file (.wav, etc.).

The file name appears in the Library page’s listbox.

Events like "app:loadFile" print to the console (debug hooks).

At this stage, playback and advanced visuals are placeholders: the UI just dispatches events so the backend (audio/visual modules) can be wired in later.

6.3 (Optional) Perlin visualizer & model integration

If you clone Nicholas’ and Aidan’s GitHub repos and put them into visualizer/ and model/, you’ll:

Install any extra requirements listed in their repos (e.g., torch, etc.).

Write glue code that:

Uses Tejas’ digital_signal and sample_rate.

Feeds audio features into the Perlin visualizer to draw images on the UI canvas.

Sends ML performance results back to the UI via the event bus ("ml:results") so the Models page can display them in the table.

For the midterm demo, you only need the pieces you actually plan to show (e.g., UI + audio input + static visual).

7. Common issues & tips

Mic not recording / permission error

Check OS privacy settings (microphone access must be enabled for the terminal / Python).

Tkinter errors on Linux

Install Tkinter separately: sudo apt install python3-tk.

Virtualenv confusion

Always confirm (.venv) is visible in the terminal before running pip install or python ....

8. One-line “How to run” summary (for the presentation slide)

Clone/download the project.

Create & activate a virtualenv.

pip install sounddevice numpy scipy matplotlib

Run:

python audio_input/tejas_audio.py to test mic / .wav input

python ui/instrument_ui.py to launch the UI
