import numpy as np
import pandas as pd
import tensorflow as tf
import os
from scipy.spatial import distance

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class HashNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class HashTable:
    def __init__(self):
        self.table = dict()

    def add(self, key, value):
        if not self.table.get(key):
            self.table[key] = list()
        self.table[key].append(HashNode(key, value))

    def find_similarity(self, value):
        min_losses = []
        for k, v in self.table.items():
            for emb in v:
                min_losses.append((self.euclidean_distance(value, emb.value), emb.key))

        return min(min_losses)

    def find_by_key(self, key):
        return self.table[key]

    @staticmethod
    def euclidean_distance(x, y):
        return distance.euclidean(x, y)


if __name__ == "__main__":
    hashtable = HashTable()
    df = pd.read_csv('db.tsv', sep='\t')
    df = df.drop(df.columns[1], axis=1)
    for row in df.itertuples():
        hashtable.add(row[1], np.array(row[2:], dtype='float64'))

    print(hashtable.find_similarity(np.array(
        [2.1682727, 2.0728548, 0.80999964, 0.47400892, -0.48876873, -5.323161, 2.5849268, -1.7018127, -4.326274, 0.6137953, -5.9668827, -0.45997593, 0.48422825, -1.0119407, -1.3262901, 1.1058555, -2.116451, 0.3111759, 0.31396174, 2.2952151, -0.7740734, 0.297204, 0.65243423, 4.027592, -8.194031, 0.4628141, -0.18898691, -2.4177494, 2.9922185, -3.4740465, -1.679268, 3.083404, -0.22320156, 1.0092336, -1.1934026, -1.3319468, 1.7131857, -0.25712672, -1.5254434, -1.2163239, -0.01938174, -0.21008573, -0.5881009, 0.8870177, 1.5702236, 2.1154354, -7.0601616, 0.25772682, -0.4016433, -1.9347364, -2.123996, -2.0914083, 0.37827432, -3.5419533, -3.6346838, 0.64074147, 1.559527, -0.35141906, 1.0389721, 2.1293306, 2.356595, 0.83773756, 3.1674426, 1.9291649, -0.37704456, -0.8969685, -2.8767948, 1.001268, -0.48483858, -1.6456615, 4.0054398, 0.5694251, -0.028162457, 2.7421625, -0.2423375, -3.8115137, -4.0642414, -2.0138733, -2.1845071, -1.0775388, 4.125232, -0.96998596, 1.0903094, -2.7385583, 0.42689273, 2.4652512, 0.80333745, 5.791132, -2.423798, -0.9181527, -4.459396, -1.4657637, 1.6288741, -2.191542, 0.41308594, 1.4317477, -1.5099578, -3.0093455, 2.7506163, 1.089634, 3.9786084, 8.127161, 2.0672023, 0.6274449, 2.2796652, -2.1234558, 1.8385462, 0.6198558, 0.5290574, -0.50677484, -1.0163866, 0.35052544, -3.141245, 9.451159, -1.8443068, -0.6214391, 2.3943036, 0.26238114, 1.6137607, -2.4635713, -5.9065037, 4.663728, -1.0760745, -0.17652604, 0.79949903, 0.041210346, -1.4221307, -2.8267539
], dtype='float64')))
