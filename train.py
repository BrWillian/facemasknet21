from model.facemasknet import FaceMaskNet
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

train_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    '/home/willian/Downloads/post-processed',
    target_size=(227, 227),
    batch_size=32,
    class_mode='sparse'
)

if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    print('Número de gpus: {}'.format(strategy.num_replicas_in_sync))

    callbacks = [ModelCheckpoint(
        'weights/facemasknet.h5',
        monitor='loss',
        save_best_only=True,
        save_freq="epoch",
        save_weights_only=False
    )]

    optimizer = Adam()

    if strategy.num_replicas_in_sync > 1:
        with strategy.scope():
            model = FaceMaskNet()
            model.summary()
            model.compile(optimizer=optimizer, loss=tfa.losses.TripletHardLoss())
    else:
        model = FaceMaskNet()
        model.summary()
        model.compile(optimizer=optimizer, loss=tfa.losses.TripletHardLoss())

    model.fit(train_generator, epochs=20, callbacks=callbacks)



