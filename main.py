import numpy as np
import os
import cv2
from model.facemasknet21 import FaceMaskNet
import tensorflow as tf


def list_dir(dir):
    data = []
    for _, _, files in os.walk(dir):
        for file in files:
            data.append(file)

    data = list(map(lambda s: s.split("")[0], data))
    return data


def train_generator(**kwargs):
    directory = kwargs.get('directory') if kwargs.get('directory') else 'input/train'
    data = sorted(list_dir(directory))
    batchSize = kwargs.get('batch_size') if kwargs.get('batch_size') else 16

    while True:
        for start in range(0, len(data), batchSize):
            x_batch = []
            end = min(start + batchSize, len(data))
            id_train_batch = data[start:end]
            for id in id_train_batch:
                im = cv2.imread(directory + '{}.jpg'.format(id))
                im = cv2.resize(im, (227, 227, 3))
                x_batch.append(im)

            x_batch = np.array(x_batch, np.float32) / 255
            yield x_batch


if __name__ == "__main__":
    model = FaceMaskNet()
    print(model.summary())
