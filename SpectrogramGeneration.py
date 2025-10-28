#authors: ir496 (Iris Robedeaux)

#import libraries
import librosa as lb
import numpy as np
import matplotlib.pyplot as plt

#set global variables
originalSampleRate = 44100
newSampleRate = 22050 

#import the data
#ONE AUDIO FOR TESTING
audioClip = lb.load("./IRMAS-Sample/Training/vio/001__[vio][nod][cou_fol]2194__1.wav")

#transform the clip for processing
spectroData = lb.feature.melspectrogram(y=audioClip[0], n_fft=1024)

#plot the spectrogram
fig, ax = plt.subplots()
S_dB = lb.power_to_db(spectroData, ref=np.max)
img = lb.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=newSampleRate,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')

#show the plot
plt.show()