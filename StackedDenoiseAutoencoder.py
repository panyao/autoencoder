#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, widgets
import numpy as np

def to_features(X):
    return X.reshape(-1, 784).astype("float32") / 255.0

def to_images(X):
    return (X*255.0).astype('uint8').reshape(-1, 28, 28)

(images_train, labels_train), (images_test, labels_test) = mnist.load_data()

X_train = to_features(images_train)
X_test = to_features(images_test)  

y_train = np_utils.to_categorical(labels_train, nb_classes=10)
y_test = np_utils.to_categorical(labels_test, nb_classes=10)


def plot_1_by_2_images(image, reconstruction, figsize=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 2, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    ax = fig.add_subplot(1, 2, 2)
    ax.matshow(reconstruction, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()
      
    
class StackedAutoencoder(object):
    
    def __init__(self, layers, mode='autoencoder',
                 activation='sigmoid', init='uniform', final_activation='softmax',
                 dropout=0.2, optimizer='SGD', metrics=["accuracy"]):
        self.layers = layers
        self.mode = mode
        self.activation = activation
        self.final_activation = final_activation
        self.init = init
        self.dropout = dropout
        self.optimizer = optimizer
        self.metrics = metrics
        self._model = None
        
        self.build()
        self.compile()
    
    def _add_layer(self, model, i, is_encoder):
        if is_encoder:
            input_dim, output_dim = self.layers[i], self.layers[i+1]
            activation = self.final_activation if i==len(self.layers)-2 else self.activation
        else:
            input_dim, output_dim = self.layers[i+1], self.layers[i]
            activation = self.activation
        model.add(Dense(output_dim=output_dim,
                        input_dim=input_dim,
                        init=self.init))
        model.add(Activation(activation))
        
    def build(self):
        self.encoder = Sequential()
        self.decoder = Sequential()
        self.autoencoder = Sequential()
        for i in range(len(self.layers)-1):
            self._add_layer(self.encoder, i, True)
            self._add_layer(self.autoencoder, i, True)
            #if i<len(self.layers)-2:
            #    self.autoencoder.add(Dropout(self.dropout))

        # Note that the decoder layers are in reverse order
        for i in reversed(range(len(self.layers)-1)):
            self._add_layer(self.decoder, i, False)
            self._add_layer(self.autoencoder, i, False)
            
    def compile(self):
        print("Compiling the encoder ...")
        self.encoder.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=self.metrics)
        print("Compiling the decoder ...")
        self.decoder.compile(loss='mse', optimizer=self.optimizer, metrics=self.metrics)
        print("Compiling the autoencoder ...")
        return self.autoencoder.compile(loss='mse', optimizer=self.optimizer, metrics=self.metrics)
    
    def fit(self, X_train, Y_train, batch_size, nb_epoch, verbose=1):
        result = self.autoencoder.fit(X_train, Y_train,
                                      batch_size=batch_size, nb_epoch=nb_epoch,
                                      verbose=verbose)
        # copy the weights to the encoder
        for i, l in enumerate(self.encoder.layers):
            l.set_weights(self.autoencoder.layers[i].get_weights())
        for i in range(len(self.decoder.layers)):
            self.decoder.layers[-1-i].set_weights(self.autoencoder.layers[-1-i].get_weights())
        return result
    
    def pretrain(self, X_train, batch_size, nb_epoch, verbose=1):
        for i in range(len(self.layers)-1):
            # Greedily train each layer
            print("Now pretraining layer {} [{}-->{}]".format(i+1, self.layers[i], self.layers[i+1]))
            ae = Sequential()
            self._add_layer(ae, i, True)
            #ae.add(Dropout(self.dropout))
            self._add_layer(ae, i, False)
            ae.compile(loss='mse', optimizer=self.optimizer, metrics=self.metrics)
            ae.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
            # Then lift the training data up one layer
            print("\nTransforming data from", X_train.shape, "to", (X_train.shape[0], self.layers[i+1]))
            enc = Sequential()
            self._add_layer(enc, i, True)
            enc.compile(loss='mse', optimizer=self.optimizer, metrics=self.metrics)
            enc.layers[0].set_weights(ae.layers[0].get_weights())
            enc.layers[1].set_weights(ae.layers[1].get_weights())
            X_train = enc.predict(X_train, verbose=verbose)
            print("\nShape check:", X_train.shape)
            # Then copy the learned weights
            self.encoder.layers[2*i].set_weights(ae.layers[0].get_weights())
            self.encoder.layers[2*i+1].set_weights(ae.layers[1].get_weights())
            self.autoencoder.layers[2*i].set_weights(ae.layers[0].get_weights())
            self.autoencoder.layers[2*i+1].set_weights(ae.layers[1].get_weights())
            self.decoder.layers[-1-(2*i)].set_weights(ae.layers[-1].get_weights())
            self.decoder.layers[-1-(2*i+1)].set_weights(ae.layers[-2].get_weights())
            self.autoencoder.layers[-1-(2*i)].set_weights(ae.layers[-1].get_weights())
            self.autoencoder.layers[-1-(2*i+1)].set_weights(ae.layers[-2].get_weights())
            
    def finetuning(self, X_train, y_train, X_test, y_test):
        model = self.encoder         
        
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test score before fine tuning:', score)
        
        model.fit(X_train, y_train, batch_size=128, nb_epoch=3, validation_data=(X_test, y_test))
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test score after fine tuning:', score)
        
    def evaluate(self, X_test, Y_test):
        return self.autoencoder.evaluate(X_test, Y_test)
    
    def predict(self, X, verbose=False):
        return self.autoencoder.predict(X, verbose=verbose)

    def _get_paths(self, name):
        model_path = "models/{}_model.yaml".format(name)
        weights_path = "models/{}_weights.hdf5".format(name)
        return model_path, weights_path

    def save(self, name='autoencoder'):
        model_path, weights_path = self._get_paths(name)
        open(model_path, 'w').write(self.autoencoder.to_yaml())
        self.autoencoder.save_weights(weights_path, overwrite=True)
    
    def load(self, name='autoencoder'):
        model_path, weights_path = self._get_paths(name)
        self.autoencoder = keras.models.model_from_yaml(open(model_path))
        self.autoencoder.load_weights(weights_path)
  
    
batch_size = 128
nb_input = 784
nb_epoch = 3

sae = StackedAutoencoder(layers=[nb_input, 400, 100, 10],
                         activation='sigmoid',
                         final_activation='sigmoid',
                         init='uniform',
                         dropout=0.2,
                         optimizer='adam')

sae.pretrain(X_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

sae.finetuning(X_train, y_train, X_test, y_test)

def draw_sae_prediction(j):
    X_plot = X_test[j:j+1]
    prediction = sae.predict(X_plot, verbose=False)
    plot_1_by_2_images(to_images(X_plot)[0], to_images(prediction)[0])
    print(sae.encoder.predict(X_plot, verbose=False)[0])

sae.evaluate(X_test, X_test)



