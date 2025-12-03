import sys
import numpy as np
import soundfile as sf
from sklearn.decomposition import NMF
import scipy.signal as signal
import matplotlib.pyplot as plt



def load_audio_mono(path):
    """
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

    for i, mag_i in enumerate(mags):
        # Soft mask to keep consistent with original mix energy
        mask = mag_i / mag_sum
        mag_masked = mag * mask

        yi = istft_from_mag_phase(mag_masked, phase, sr, n_fft=n_fft, hop_length=hop)

        out_path = f"{output_prefix}_source{i+1}.wav"
        # soundfile will write float32 WAV by default
        sf.write(out_path, yi, sr, subtype="PCM_16")
        print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    # EITHER: python source_separation.py input.wav output_prefix
    if len(sys.argv) == 3:
        in_path = sys.argv[1]
        out_prefix = sys.argv[2]
        separate_file(in_path, out_prefix)
    else:
        # Hardcoded quick test
        separate_file(
            r"C:\Users\13523\Desktop\TimbreMind\trimmed_clip.wav",
            r"C:\Users\13523\Desktop\TimbreMind\out",
        )

