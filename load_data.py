import os, glob
import numpy as np

from sklearn.model_selection import train_test_split

from extract_feature import extract_feature


def load_data(base_file_path, emotions, observed_emotions, test_size=0.2):
    x,y=[],[]
    file_path = base_file_path + '/Actor_*/*.wav'

    raw_files = glob.glob(file_path)

    if not raw_files:
        raise RuntimeError('No .wav files found in %s' % file_path)

    for raw_file in raw_files:
        file_name=os.path.basename(raw_file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(raw_file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)