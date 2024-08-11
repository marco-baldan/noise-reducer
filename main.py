import librosa
import numpy as np
import soundfile as sf

# Load the audio file
y, sr = librosa.load('mywav.wav', sr=None)

stft = librosa.stft(y)

magnitude, phase = np.abs(stft), np.angle(stft)

noise_frames = magnitude[:, :10]
noise_profile = np.mean(noise_frames, axis=1, keepdims=True)

magnitude_denoised = np.maximum(magnitude - noise_profile, 0)

stft_denoised = magnitude_denoised * np.exp(1j * phase)

y_denoised = librosa.istft(stft_denoised)

# Save the noise-reduced audio back to a file
sf.write('mywav_reduced_noise.wav', y_denoised, sr)

print("Noise reduction complete. File saved as 'mywav_reduced_noise.wav'.")
