import numpy as np
import matplotlib.pyplot as plt

# Example: create a complex sinusoidal signal with phase shifts
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs)
signal = np.exp(1j * (2 * np.pi * 5 * t + np.pi * (t > 0.5)))  # Phase shift at t=0.5

phase = np.angle(signal)  # Instantaneous phase [-π, π]
unwrapped_phase = np.unwrap(phase)
phase_shifts = np.diff(unwrapped_phase)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(t, unwrapped_phase)
plt.title("Unwrapped Phase")
plt.xlabel("Time [s]")

plt.subplot(1, 2, 2)
plt.plot(t[1:], phase_shifts)
plt.title("Phase Shifts")
plt.xlabel("Time [s]")

plt.tight_layout()
plt.show()