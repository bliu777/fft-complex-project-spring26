import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fft
from scipy import signal
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

#Read in .wav file as data
(rate, data) = wavfile.read('./sampleAudio.wav')
t_x = np.arange(data.size)*(1/rate)
N = data.size
print(rate, data.size, data.size/rate)

#Apply FFT
#fftdata = fft.fft(data)
#print(fftdata)
w = gaussian(20,3)
sft = ShortTimeFFT(w, hop=5, fs=1/rate, mfft=40, scale_to="magnitude")
sx = sft.stft(data)
#print(abs(sx))

#Plot FFT
# freqs = fft.fftfreq(data.size, 1./rate)
# print(fft.fftshift(freqs))
# plt.plot(freqs, fftdata.real, label="real component")
# plt.plot(freqs, fftdata.imag, label="imaginary component")
# plt.legend()
# plt.show()
# fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
# t_lo, t_hi = sft.extent(N)[:2]  # time range of plot
# ax1.set_title(rf"STFT ({sft.m_num*sft.T:g}$\,s$ Gaussian window, " +
#               rf"$\sigma_t={3*sft.T}\,$s)")
# ax1.set(xlabel=f"Time $t$ in seconds ({sft.p_num(N)} slices, " +
#                rf"$\Delta t = {sft.delta_t:g}\,$s)",
#         ylabel=f"Freq. $f$ in Hz ({sft.f_pts} bins, " +
#                rf"$\Delta f = {sft.delta_f:g}\,$Hz)",
#         xlim=(t_lo, t_hi))

# im1 = ax1.imshow(abs(sx), origin='lower', aspect='auto',
#                  extent=sft.extent(N), cmap='viridis')
# fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

# # Shade areas where window slices stick out to the side:
# for t0_, t1_ in [(t_lo, sft.lower_border_end[0] * sft.T),
#                  (sft.upper_border_begin(N)[0] * sft.T, t_hi)]:
#     ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
# for t_ in [0, N * sft.T]:  # mark signal borders with vertical line:
#     ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)
# ax1.legend()
# fig1.tight_layout()
# plt.show()

#Apply low-pass filter
sos = signal.butter(2, 0.2, output="sos")
sxfiltered = signal.sosfilt(sos, sx)

#Invese Transform
filterData = sft.istft(sxfiltered, k1=data.size)
filterData = np.int16(filterData / np.max(np.abs(filterData))* 32767)

#Convert back into .wav file
wavfile.write("filteredAudio.wav", rate, filterData)