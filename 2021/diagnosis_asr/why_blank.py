import os

import bob.io.audio
import bob.kaldi
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt


def compute_vad_percent(filename):
    data = bob.io.audio.reader(filename)
    vad = bob.kaldi.compute_dnn_vad(data.load()[0], data.rate)
    ones = np.count_nonzero(vad)
    return ones / len(vad)


blank_vad = []
not_blank_vad = []
blank_duration = []
not_blank_duration = []
blank_energy = []
not_blank_energy = []
blank_files = os.listdir('./blanks')
for file in blank_files:
    try:
        filepath = os.path.join("./blanks", file)
        rate, audData = scipy.io.wavfile.read(filepath)
        duration = len(audData) / float(rate)
        energy = np.sum(audData.astype(float) ** 2)
        vad = compute_vad_percent(filepath)
        blank_duration.append(duration)
        blank_energy.append(energy)
        blank_vad.append(vad)
    except:
        pass

for file in os.listdir('./tmp'):
    if file in os.listdir('./blanks'):
        continue
    try:
        filepath = os.path.join("./tmp", file)
        rate, audData = scipy.io.wavfile.read(filepath)
        duration = len(audData) / float(rate)
        energy = np.sum(audData.astype(float) ** 2)
        vad = compute_vad_percent(filepath)
        not_blank_duration.append(duration)
        not_blank_energy.append(energy)
        not_blank_vad.append(vad)
    except:
        pass

# plt.plot(sorted(blank_energy))
# plt.plot(sorted(not_blank_energy))
# plt.show()
# plt.plot(sorted(blank_duration))
# plt.plot(sorted(not_blank_duration))
# plt.show()
plt.plot(sorted(blank_vad))
plt.plot(sorted(not_blank_vad))
plt.savefig('vad.png')
print(np.mean(blank_vad), np.std(blank_vad))
print(np.mean(not_blank_vad), np.std(not_blank_vad))
# sns.histplot(data=blank_energy, bins=50)
# plt.show()
# sns.histplot(data=not_blank_energy, bins=50)
# plt.show()
