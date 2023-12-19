import joblib
import numpy as np
import tensorflow as tf


class PointsClassifier:
    def __init__(self):
        # Load the model parameters
        self.model_params = {
            'layer_sizes': [400, 200, 100, 100],
            'dropout_rate': 0.5,
            'epsilon': 0.001,
            }
        self.data_params = {
            'n_points': 20,
            }
        self.categories = ['cF', 'cP', 'cI', 'hP', 'hR', 'mC', 'mP', 'oP', 'oC', 'oF', 'oI', 'tI', 'tP', 'aP']
        self.n_categories = len(self.categories)
        inputs = tf.keras.Input(shape=self.data_params['n_points'], name='input_points')
        self.model = tf.keras.Model(inputs, self.model_builder(inputs))
        self.model.load_weights(
            filepath='classification_model_weights.h5',
            by_name=True
            )
        self.points_transformer = joblib.load('classification_points_transformer.bin')

    def model_builder(self, x):
        for index in range(len(self.model_params['layer_sizes'])):
            x = tf.keras.layers.Dense(
                self.model_params['layer_sizes'][index],
                activation=None,
                name='dense_' + str(index)
                )(x)
            x = tf.keras.layers.BatchNormalization(
                epsilon=self.model_params['epsilon'], 
                name='batch_norm_' + str(index)
                )(x)
            x = tf.keras.layers.LayerNormalization(
                epsilon=self.model_params['epsilon'], 
                name='layer_norm_' + str(index)
                )(x)
            x = tf.keras.activations.gelu(x)
            x = tf.keras.layers.Dropout(
                rate=self.model_params['dropout_rate'],
                name=f'dropout_{index}'
                )(x)
        x = tf.keras.layers.Dense(
            self.n_categories,
            activation='softmax',
            name='classification_softmax',
            )(x)
        return x

    def do_inference(self, points):
        points_transformed = self.points_transformer.transform(points[:, np.newaxis])[:, 0]
        softmaxes = self.model.predict(points_transformed[np.newaxis, :])
        maximum_index = np.argmax(softmaxes, axis=1)[0]
        print()
        print('Interpreting softmaxes as probabilities')
        print(f'Most Likely Class: {self.categories[maximum_index]} with probability {softmaxes[0, maximum_index]:0.2f}')
        print()
        print('Other probabilities:')
        for index in range(self.n_categories):
            print(f'  {self.categories[index]}: {softmaxes[0, index]:0.2f}')

if (__name__ == '__main__'):
    classifier = PointsClassifier()
    classifier.do_inference(np.arange(1, 21))
