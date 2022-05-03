import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.facemasknet import FaceMaskNet

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class SaveBaseModel(object):
    '''
        Classe responsavel por salvar toda a base de dados contendo as matricas de cada face.
    '''
    def __init__(self, **kwargs):
        super(SaveBaseModel, self).__init__()
        self._directory = kwargs.get("directory") if kwargs.get("directory") else "dataset/"
        self._model_path = kwargs.get("path_model") if kwargs.get("path_model") else "weights/facemasknet.h5"
        self._train_generator = self._load_datagen()
        self._model = FaceMaskNet()
        self._classes_names = []
        self._db = []

    def _load_datagen(self):
        '''
            Carregar a base de imagens.
        '''

        train_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            self._directory,
            target_size=(227, 227),
            batch_size=1,
            class_mode='sparse',
            shuffle=False
        )

        return train_generator

    def get_db_embeddings(self, **kwargs):
        '''
            Faz o predição das imagens, carrega e rotula todo o dataset, transforma para dataframe.
        '''
        classes_names = self._get_class_name()
        predicted = self._load_model_facemasknet()

        for _class, _embeddings in zip(self._train_generator, predicted):
            _class = list(_class[1])
            _embeddings = list(_embeddings)
            self._db.append([classes_names[int(*_class)], int(*_class), *_embeddings])

        df = pd.DataFrame(self._db)

        if kwargs.get("save"):
            df.to_csv('db.tsv', sep='\t', index=False)

        return df

    def _get_class_name(self):
        '''
            Pega todos os nomes baseadas em pastas.
        '''
        for _, dir, _ in os.walk(self._directory):
            for path in sorted(dir):
                self._classes_names.append(path)

        return self._classes_names

    def _load_model_facemasknet(self):
        '''
            Carrega o modelo e faz a predição.
        '''
        self._model.load_weights(filepath=self._model_path)

        results = self._model.predict(self._train_generator)

        return results


if __name__ == "__main__":
    SaveBaseModel = SaveBaseModel(directory='/home/willian/Downloads/post-processed/', path_model='weights/facemasknet.h5')
    SaveBaseModel.get_db_embeddings(save=True)
