from abc import ABC

from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalMaxPool2D, Dense, BatchNormalization, Input
from tensorflow.keras.models import Model
import tensorflow as tf


class FaceMaskNet():
    def __init__(self):
        super(FaceMaskNet, self).__init__()
        self.model = self.get_model()

    def get_model(self):
        inputs = Input(shape=(227, 227, 3))
        x = Conv2D(64, 3, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(128, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((4, 4))(x)
        x = Conv2D(256, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, 3, activation='relu')(x)
        x = GlobalMaxPool2D()(x)
        output = Dense(128)(x)
        return Model(inputs=[inputs], outputs=[output])

    def __getattr__(self, name):
        return getattr(self.model, name)
