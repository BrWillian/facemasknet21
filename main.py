import numpy as np
import io
from model.facemasknet import FaceMaskNet
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa

train_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    '/home/willian/Downloads/post-processed',
    target_size=(227, 227),
    batch_size=64,
    class_mode='sparse'
)

if __name__ == "__main__":

    strategy = tf.distribute.MirroredStrategy()
    print('Número de gpus: {}'.format(strategy.num_replicas_in_sync))

    if strategy.num_replicas_in_sync > 1:
        with strategy.scope():
            model = FaceMaskNet()
            model.summary()
            model.compile(optimizer='Adam', loss=tfa.losses.TripletHardLoss())
    else:
        model = FaceMaskNet()
        model.summary()
        model.compile(optimizer='Adam', loss=tfa.losses.TripletHardLoss())

    model.fit(train_generator, epochs=1)

    results = model.predict(train_generator)

    np.savetxt("vetor.tsv", results, delimiter='\t')

    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    for _, labels in train_generator:
        [out_m.write(str(x)) for x in labels]
    out_m.close()



