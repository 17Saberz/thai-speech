import librosa
import os

# filepath = 'D:\\Tech1\\dataset\\speaker01\\correct\\test0.wav'
filepath = 'D:\Tech1\dataset\speaker04\correct\k_correct_normal_round5.wav'

y, sr = librosa.load(filepath, sr=None)

iterval = librosa.effects.split(y, top_db=25)

print("Speech Segements (sec) :")
for start, end in iterval:
    print(f"{start/sr:.2f} - {end/sr:.2f} sec")