import numpy as np
from scipy.io import wavfile
import glob
from mido import MidiFile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

SAMPLE_RATE = 44100

NOTE_FREQUENCIES = { #freqs in the 4th octave
    'C': 261.63,
    'C#': 277.18,
    'D': 293.66,
    'D#': 311.13,
    'E': 329.63,
    'F': 349.23,
    'F#': 369.99,
    'G': 392.00,
    'G#': 415.30,
    'A': 440.00,
    'A#': 466.16,
    'B': 493.88
}

def note_name_to_frequency(note_name):
    # Q1: Your code goes here
    note = note_name[:-1]
    octave = int(note_name[-1])
    frequency = 0.0

    if octave == 4:
        frequency = NOTE_FREQUENCIES[note]
    elif octave > 4:
        frequency = NOTE_FREQUENCIES[note] * (2 ** (octave - 4))
    else:
        frequency = NOTE_FREQUENCIES[note] / (2 ** (4 - octave))

    return frequency

# print(note_name_to_frequency('A4'),  # Example usage
# note_name_to_frequency('C3'),
# note_name_to_frequency('G5'))

def decrease_amplitude(audio):
    # Q2: Your code goes here

    return audio * np.linspace(1, 0, len(audio))

def add_delay_effects(audio):
    #Q3: Your code goes here
    delay_seconds = 0.5
    delay_samples = int(delay_seconds * SAMPLE_RATE)

    # output length is longer to accommodate the delayed signal.
    output_length = len(audio) + delay_samples
    delayed_audio = np.zeros(output_length)
    
    # Add the original audio at 70% amplitude.
    delayed_audio[:len(audio)] += 0.7 * audio
    # Add the delayed audio at 30% amplitude.
    delayed_audio[delay_samples:] += 0.3 * audio

    return delayed_audio

def concatenate_audio(list_of_your_audio):
    #Q4: Your code goes here
    return np.concatenate(list_of_your_audio)

def mix_audio(list_of_your_audio, amplitudes):
    #Q4: Your code goes here
    mixed = np.zeros_like(list_of_your_audio[0])
    for audio, amp in zip(list_of_your_audio, amplitudes):
        mixed += amp * audio
    return mixed

def create_sawtooth_wave(frequency, duration, sample_rate=44100):
    #Q5: Your code goes here
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    wave = np.zeros_like(t)
    for k in range(1, 20):
        wave += ((-1)**(k+1) / k) * np.sin(2 * np.pi * k * frequency * t)
    wave *= (2 / np.pi)
    return wave

def get_file_lists():
    piano_files = sorted(glob.glob("./piano/*.mid"))
    drum_files = sorted(glob.glob("./drums/*.mid"))
    return piano_files, drum_files

def get_num_beats(file_path):
    # Q6: Your code goes here
    mid = MidiFile(file_path)
    # Might need: mid.tracks, msg.time, mid.ticks_per_beat
    total_ticks = 0
    # calculate the cumulative delta time for each track and choose the maximum
    for track in mid.tracks:
        track_ticks = sum(msg.time for msg in track)
        if track_ticks > total_ticks:
            total_ticks = track_ticks
    # Number of beats (ensure ticks_per_beat is non-zero)
    nBeats = 0
    
    if mid.ticks_per_beat:
        nBeats = total_ticks / mid.ticks_per_beat

    return nBeats

def get_stats(piano_path_list, drum_path_list):
    piano_beat_nums = []
    drum_beat_nums = []
    for file_path in piano_path_list:
        piano_beat_nums.append(get_num_beats(file_path))
        
    for file_path in drum_path_list:
        drum_beat_nums.append(get_num_beats(file_path))
    
    return {"piano_midi_num":len(piano_path_list),
            "drum_midi_num":len(drum_path_list),
            "average_piano_beat_num":np.average(piano_beat_nums),
            "average_drum_beat_num":np.average(drum_beat_nums)}