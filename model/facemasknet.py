from tensorflow.keras.layers import Conv2D, Lambda, MaxPool2D, Flatten, Dense, BatchNormalization, Input, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf


class FaceMaskNet():
    def __init__(self):
        super(FaceMaskNet, self).__init__()
        self.model = self.get_model()

    def get_model(self):
        #Block 1
        inputs = Input(shape=(227, 227, 3))
        x = Conv2D(96, 11, activation='relu', strides=(4, 4))(inputs)
        x = BatchNormalization()(x)
        x = MaxPool2D((3, 3), strides=(2, 2))(x)
        x = Dropout(0.3)(x)

        #Block 2
        x = Conv2D(96, 5, activation='relu', strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((3, 3), strides=(2, 2))(x)
        x = Dropout(0.3)(x)

        #Block 3
        x = Conv2D(384, 3, activation='relu', padding='same', strides=(1, 1))(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((3, 3), strides=(2, 2))(x)
        x = Dropout(0.3)(x)

        #Block 4
        x = Flatten()(x)
        x = Dense(100)(x)
        x = Dense(100)(x)
        output = Dense(128, activation=None)(x)
        #output = Lambda(lambda k: tf.math.l2_normalize(k, axis=1))(x)
        return Model(inputs=[inputs], outputs=[output])

    def __getattr__(self, name):
        return getattr(self.model, name)
