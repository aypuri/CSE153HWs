#dataroot = "/home/aypuri/Documents/CSE153HWs-1/"
dataroot = "."

import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import random
import glob

torch.use_deterministic_algorithms(True) # Try to make things less random, though not required

audio_paths = glob.glob(dataroot + "/nsynth_subset/*.wav")
random.seed(0)
random.shuffle(audio_paths)



if not len(audio_paths):
    print("You probably need to set the dataroot folder correctly")

SAMPLE_RATE = 8000 
N_MFCC = 13
INSTRUMENT_MAP = {'guitar': 0, 'vocal': 1} 
NUM_CLASSES = len(INSTRUMENT_MAP)

# INSTRUMENT_MAP_7 = {
#     'guitar_acoustic': 0,
#     'guitar_electronic': 1,
#     'vocal_acoustic': 2,
#     'vocal_synthetic': 3
# }


def extract_waveform(path: str) -> np.ndarray:
    # Your code here
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE)
    return waveform


def extract_label(path: str) -> int:
    # Your code here
    fname = path.split('/')[-1]
    instr = fname.split('_')[0]
    return INSTRUMENT_MAP[instr]

waveforms = [extract_waveform(p) for p in audio_paths]
labels = [extract_label(p) for p in audio_paths]

class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(2 * N_MFCC, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(nnF.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nnF.relu(self.bn2(self.conv2(x))))
        x = self.pool3(nnF.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def extract_mfcc(w: np.ndarray) -> torch.FloatTensor:
    # Your code here
    mfcc = librosa.feature.mfcc(y=w, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    means = mfcc.mean(axis=1)
    stds  = mfcc.std(axis=1)
    features = np.concatenate([means, stds], axis=0)
    return torch.from_numpy(features).float()



def extract_spec(w: np.ndarray) -> torch.FloatTensor:
    # Your code here

    D = librosa.stft(w)
    spec = np.abs(D) ** 2
    return torch.from_numpy(spec).float()


def extract_mel(w, n_mels = 128, hop_length = 512):
    # Your code here
    # load
    # convert to db
    # normalize

    S = librosa.feature.melspectrogram(
        y=w,
        sr=SAMPLE_RATE,
        n_mels=n_mels,
        hop_length=hop_length
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_min, S_max = S_db.min(), S_db.max()
    S_norm = (S_db - S_min) / (S_max - S_min + 1e-6)
    return torch.from_numpy(S_norm).float()


def extract_q(w: np.ndarray) -> torch.FloatTensor:
    # Your code here

    w16 = librosa.resample(w, orig_sr=SAMPLE_RATE, target_sr=16000)
    C = np.abs(
            librosa.cqt(
                y=w,
                sr=16000,
                n_bins=84,
                bins_per_octave=12,
                fmin=librosa.note_to_hz('C1')
            )
        )

    return torch.from_numpy(C).float()

def pitch_shift(w: np.ndarray, n: int) -> np.ndarray:
    return librosa.effects.pitch_shift(w, sr=SAMPLE_RATE, n_steps=n)

augmented_waveforms = []
augmented_labels = []

for w,y in zip(waveforms,labels):
    augmented_waveforms.append(w)
    augmented_waveforms.append(pitch_shift(w,1))
    augmented_waveforms.append(pitch_shift(w,-1))
    augmented_labels += [y,y,y]


INSTRUMENT_MAP_7 = {'guitar_acoustic': 0, 'guitar_electronic': 1, 'vocal_acoustic': 2, 'vocal_synthetic': 3}

NUM_CLASSES_7 = 4

def extract_label_7(path: str) -> int:
    # Your code here
    fname = path.split('/')[-1].replace('.wav', '')
    parts = fname.split('_')
    instr = '_'.join(parts[:2])        
    return INSTRUMENT_MAP_7[instr]


labels_7 = [extract_label_7(p) for p in audio_paths]


class MLPClassifier_7classes(torch.nn.Module):
    def __init__(self):
        super(MLPClassifier_7classes, self).__init__()
        self.fc1 = torch.nn.Linear(2 * 13, 64) 
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, NUM_CLASSES_7)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
feature_func_7 = extract_mel

class MelCNN4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.adapt = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(64, NUM_CLASSES_7)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,n_mels,frames)
        x = self.pool(nnF.relu(self.bn1(self.conv1(x))))
        x = self.pool(nnF.relu(self.bn2(self.conv2(x))))
        x = self.adapt(nnF.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model_7 = MelCNN4()