#authors: ir496 (Iris Robedeaux)

#import libraries
import librosa as lb
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

#set global variables
originalSampleRate = 44100
newSampleRate = 22050 
verbose = True

#import the data
#irmas = md.initialize("irmas", data_home="./IRMAS-Data")

## only run once (download data)
#irmas.download(cleanup = False)

#make the spectrogram folders
trainAudioPath = os.path.join('.', 'IRMAS-Data', 'IRMAS-TrainingData')
trainSpectPath = os.path.join('.', 'IRMAS-Data', 'IRMAS-Data-Training-Spec')

test1AudioPath = os.path.join('.', 'IRMAS-Data', 'IRMAS-TestingData-Part1')
test1SpectPath = os.path.join('.', 'IRMAS-Data', 'IRMAS-Test-Part1-Spec')

test2AudioPath = os.path.join('.', 'IRMAS-Data', 'IRMAS-TestingData-Part2')
test2SpectPath = os.path.join('.', 'IRMAS-Data', 'IRMAS-Test-Part2-Spec')

test3AudioPath = os.path.join('.', 'IRMAS-Data', 'IRMAS-TestingData-Part3')
test3SpectPath = os.path.join('.', 'IRMAS-Data', 'IRMAS-Test-Part3-Spec')

#loop through all of the data (test and train)
src_root = './IRMAS-Data/IRMAS-TrainingData'
dst_root = './IRMAS-Data/IRMAS-Training-Spec'

for root, dirs, files in os.walk(src_root):
    # Determine the relative path (for each instrument)
    rel_path = os.path.relpath(root, src_root)

    # Build matching destination folder
    dst_dir = os.path.join(dst_root, rel_path)
    os.makedirs(dst_dir, exist_ok=True)

    #print where we are
    if verbose == True:
        print( f"\nProcessing files in folder {rel_path}" )

    #set iterator for verbose output
    i = 1

    #loop through folder
    for file in files:

        # Only process wav files
        if not file.lower().endswith(".wav"):
            continue

        #preserve structure for spectrogram folder
        src_file = os.path.join(root, file)
        #rename file to appropriate name (png, spec)
        dst_file = os.path.join(dst_dir, file.replace('.wav', '_spec.png'))

        #load the data
        audioClip, sr = lb.load(src_file, sr=newSampleRate)

        #make a spectrogram
        spectroData = lb.feature.melspectrogram(y=audioClip, n_fft=1024)

        # Compute dB spec
        S_dB = librosa.power_to_db(spectroData, ref=np.max)

        fig, ax = plt.subplots()

        # Draw spectrogram
        img = librosa.display.specshow(
            S_dB,
            sr=newSampleRate,
            x_axis=None,
            y_axis=None,
            fmax=8000,
            ax=ax
        )

        ax.set_axis_off()               # Hide axes entirely
        plt.tight_layout(pad=0)         # Remove padding

        # Save with no whitespace
        plt.savefig(dst_file, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        if verbose == True:
            print(f"\rProcessed file {i}", end="")

        #increase iterator
        i += 1