#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_beats_resources.py
Generează graficele, fișierele audio și o animație GIF pentru fenomenul de bătăi acustice.
Rulare:
    python generate_beats_resources.py
Output:
    beats_project_output/
        figures/short_window_signals.png
        figures/beats_and_envelope.png
        figures/spectrum.png
        audio/tone_f1.wav
        audio/tone_f2.wav
        audio/beat_signal.wav
        video/beats_animation.gif
"""
import os, wave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ----------------- Parametri editabili -----------------
fs = 44100          # rată de eșantionare [Hz]
duration = 6.0      # durată [s]
f1 = 440.0          # frecvența 1 [Hz]
f2 = 444.0          # frecvența 2 [Hz]  -> f_batai = |f2 - f1|

# ----------------- Pregătire directoare -----------------
project_dir = "beats_project_output"
fig_dir = os.path.join(project_dir, "figures")
audio_dir = os.path.join(project_dir, "audio")
video_dir = os.path.join(project_dir, "video")
for d in (fig_dir, audio_dir, video_dir):
    os.makedirs(d, exist_ok=True)

# ----------------- Semnale -----------------
t = np.linspace(0, duration, int(fs*duration), endpoint=False)
x1 = np.cos(2*np.pi*f1*t)
x2 = np.cos(2*np.pi*f2*t)
x_sum = 0.5*(x1 + x2)    # factor 0.5 ca să evităm clipping în WAV
delta_f = abs(f2 - f1)
envelope = np.abs(np.cos(np.pi*delta_f*t))

# ----------------- Utilitare -----------------
def write_wav(path, data, sr):
    data_clip = np.clip(data, -1.0, 1.0)
    int_data = (data_clip * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sr)
        wf.writeframes(int_data.tobytes())

# ----------------- Audio -----------------
write_wav(os.path.join(audio_dir, "tone_f1.wav"), 0.9*x1, fs)
write_wav(os.path.join(audio_dir, "tone_f2.wav"), 0.9*x2, fs)
write_wav(os.path.join(audio_dir, "beat_signal.wav"), 0.9*x_sum, fs)

# ----------------- Grafice -----------------
# 1) Fereastră scurtă (50 ms) pentru a vedea purtătoarea
idx_short = int(0.05 * fs)
plt.figure()
plt.plot(t[:idx_short], x1[:idx_short], label=f"x1: {f1:.1f} Hz")
plt.plot(t[:idx_short], x2[:idx_short], label=f"x2: {f2:.1f} Hz")
plt.plot(t[:idx_short], x_sum[:idx_short], label="x = 0.5(x1+x2)")
plt.xlabel("t [s]"); plt.ylabel("Amplitudine"); plt.title("50 ms – comparare semnale")
plt.legend()
plt.savefig(os.path.join(fig_dir, "short_window_signals.png"), dpi=200, bbox_inches="tight")
plt.close()

# 2) Bătăi + anvelopă (2 s)
idx_long = int(2.0 * fs)
plt.figure()
plt.plot(t[:idx_long], x_sum[:idx_long], label="x(t)")
plt.plot(t[:idx_long], envelope[:idx_long], linestyle="--", label="Envelopă teoretică")
plt.plot(t[:idx_long], -envelope[:idx_long], linestyle="--")
plt.xlabel("t [s]"); plt.ylabel("Amplitudine"); plt.title(f"Bătăi acustice (Δf = {delta_f:.1f} Hz)")
plt.legend()
plt.savefig(os.path.join(fig_dir, "beats_and_envelope.png"), dpi=200, bbox_inches="tight")
plt.close()

# 3) Spectru (FFT)
Nfft = 1<<16
X = np.fft.rfft(x_sum[:Nfft], n=Nfft)
freqs = np.fft.rfftfreq(Nfft, d=1/fs)
amp = np.abs(X)/np.max(np.abs(X))
plt.figure()
plt.plot(freqs, amp)
plt.xlim(min(f1, f2)-20, max(f1, f2)+20)
plt.xlabel("Frecvență [Hz]"); plt.ylabel("Amplitudine normalizată"); plt.title("Spectrul semnalului (două linii apropiate)")
plt.savefig(os.path.join(fig_dir, "spectrum.png"), dpi=200, bbox_inches="tight")
plt.close()

# ----------------- Animație GIF (rapidă) -----------------
gif_path = os.path.join(video_dir, "beats_animation.gif")
frames = 40               # mai puține cadre pentru viteză
window = int(0.05 * fs)   # 50 ms fereastră glisantă
step = max(1, (len(t)-window)//frames)

fig, ax = plt.subplots(figsize=(4.8, 2.7), dpi=100)
line_sum, = ax.plot([], [])
line_env, = ax.plot([], [], linestyle="--")
line_envm, = ax.plot([], [], linestyle="--")
ax.set_xlabel("t [s]"); ax.set_ylabel("Amplitudine"); ax.set_title("Animație – bătăi acustice")
ax.set_xlim(0, window/fs); ax.set_ylim(-1.2, 1.2)

def init():
    line_sum.set_data([], []); line_env.set_data([], []); line_envm.set_data([], [])
    return line_sum, line_env, line_envm

def update(frame):
    i = frame*step
    j = i+window if i+window <= len(t) else len(t)
    i = i if j-i == window else len(t)-window
    seg_t = t[i:j] - t[i]
    line_sum.set_data(seg_t, x_sum[i:j])
    line_env.set_data(seg_t, envelope[i:j])
    line_envm.set_data(seg_t, -envelope[i:j])
    return line_sum, line_env, line_envm

anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
anim.save(gif_path, writer=PillowWriter(fps=15))
plt.close(fig)

print("Gata. Resurse în", project_dir)
