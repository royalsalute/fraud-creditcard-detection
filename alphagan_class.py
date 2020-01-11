from __future__ import print_function, division

import os

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import numpy as np


class AlphaGAN():
    def __init__(self, lambda_=1, lr1=0.0005, lr2=0.0001, beta1=0.5, beta2=0.9, model_save_path="./snapshots"):
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        self.model_save_path = model_save_path

        self.input_dim = 29
        self.x_shape = (self.input_dim, )
        self.latent_dim = 16
        self.base_n_count = 128
        self.lambda_ = lambda_

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        optimizer_disc = Adam(learning_rate=lr2, beta_1=beta1, beta_2=beta2)
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer_disc, metrics=['accuracy'])

        # Build and compile the codecriminator
        self.code_discriminator = self.build_code_discriminator()
        optimizer_code_disc = Adam(learning_rate=lr2, beta_1=beta1, beta_2=beta2)
        self.code_discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer_code_disc, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Build the encoder
        self.encoder = self.build_encoder()

        # The part of the alphagan that trains the discriminator and encoder
        self.discriminator.trainable = False
        self.code_discriminator.trainable = False

        # Generate image from sampled noise
        z_rand = Input(shape=(self.latent_dim,))
        x_rand = self.generator(z_rand)

        # Encode image
        x = Input(shape=self.x_shape)
        z_hat = self.encoder(x)
        x_hat = self.generator(z_hat)

        # Latent -> img is fake, and img -> latent is valid
        fake_d = self.discriminator(x_rand)
        valid_d = self.discriminator(x_hat)

        # code discriminator
        # fake_c = self.code_discriminator(z_rand)
        valid_c = self.code_discriminator(z_hat)

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.alphagan_generator = Model(inputs=[z_rand, x], outputs=[fake_d, valid_d, valid_c, x_hat])
        self.alphagan_generator.compile(
            loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'mean_absolute_error'],
            loss_weights=[1., 1., 1., self.lambda_],
            optimizer=Adam(learning_rate=lr1, beta_1=beta1, beta_2=beta2))

    def build_encoder(self):
        model = Sequential(name="Encoder")

        model.add(Dense(self.base_n_count * 2))
        model.add(ReLU())
        model.add(Dense(self.base_n_count))
        model.add(ReLU())
        model.add(Dense(self.latent_dim))
        model.add(Activation('tanh'))

        x = Input(shape=self.x_shape)
        z = model(x)
        model.summary()

        return Model(x, z)

    def build_generator(self):
        model = Sequential(name="Generator")

        model.add(Dense(self.base_n_count))
        model.add(ReLU())
        model.add(Dense(self.base_n_count * 2))
        model.add(ReLU())
        model.add(Dense(self.base_n_count * 4))
        model.add(ReLU())
        model.add(Dense(self.input_dim))
        model.add(Activation('tanh'))

        z = Input(shape=(self.latent_dim,))
        x_gen = model(z)
        model.summary()

        return Model(z, x_gen)

    def build_discriminator(self):
        model = Sequential(name="Discriminator")

        model.add(Dense(self.base_n_count * 4))
        model.add(LeakyReLU())
        model.add(Dense(self.base_n_count * 2))
        model.add(LeakyReLU())
        model.add(Dense(self.base_n_count))
        model.add(LeakyReLU())
        model.add(Dense(1, activation='sigmoid'))

        x = Input(shape=self.x_shape)
        validity = model(x)
        model.summary()

        return Model(x, validity)

    def build_code_discriminator(self):
        model = Sequential(name="CodeDiscriminator")

        model.add(Dense(self.base_n_count * 4))
        model.add(LeakyReLU())
        model.add(Dense(self.base_n_count * 2))
        model.add(LeakyReLU())
        model.add(Dense(self.base_n_count))
        model.add(LeakyReLU())
        model.add(Dense(1, activation='sigmoid'))

        z = Input(shape=(self.latent_dim,))
        validity = model(z)
        model.summary()

        return Model(z, validity)

    def train(self, X_train, epochs, batch_size=32, output_path='.', model_save_step=100):
        # if not os.path.exists(os.path.join(output_path, 'logs/')):
        #     os.makedirs(os.path.join(output_path, 'logs/'))

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # Sample noise and generate img
            z_rand = np.random.normal(size=(batch_size, self.latent_dim))
            x_rand = self.generator.predict(z_rand)

            # Select a random batch of images and encode
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            x = X_train[idx]
            z_hat = self.encoder.predict(x)
            x_hat = self.generator.predict(z_hat)

            # Train the code discriminator
            c_hat_loss = self.code_discriminator.train_on_batch(z_hat, valid)
            c_fake_loss = self.code_discriminator.train_on_batch(z_rand, fake)
            c_loss = np.add(c_hat_loss, c_fake_loss)

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch(x, valid)
            d_loss_rec = self.discriminator.train_on_batch(x_hat, fake)
            d_loss_fake = self.discriminator.train_on_batch(x_rand, fake)
            d_loss = np.add(np.add(d_loss_real, d_loss_rec), d_loss_fake)

            # Train the generator (z -> img is valid and img -> z is is invalid), encoder
            g_loss = self.alphagan_generator.train_on_batch([z_rand, x], [valid, valid, valid, x])

            # Plot the progress
            if epoch % 100 == 0:
                print("%d [D loss: %f, acc: %.2f%%] [C loss: %f, acc: %.2f%%] [G loss: %f]" % \
                      (epoch, d_loss[0], 100 * d_loss[1] / 3, c_loss[0], 100 * c_loss[1] / 2, g_loss[0]))

            if (epoch + 1) % model_save_step == 0:
                self.generator.save(os.path.join(self.model_save_path, '{}_G.h5'.format(epoch + 1)))
                self.encoder.save(os.path.join(self.model_save_path, '{}_E.h5'.format(epoch + 1)))
                self.discriminator.save(os.path.join(self.model_save_path, '{}_D.h5'.format(epoch + 1)))
                self.code_discriminator.save(os.path.join(self.model_save_path, '{}_C.h5'.format(epoch + 1)))

    def load_pretrained_models(self, model_path_prefix):
        self.generator.load_weights('%sG.h5' % model_path_prefix)
        self.encoder.load_weights('%sE.h5' % model_path_prefix)
        self.discriminator.load_weights('%sD.h5' % model_path_prefix)
        self.code_discriminator.load_weights('%sC.h5' % model_path_prefix)


# if __name__ == '__main__':
#     alphagan = AlphaGAN()
#     alphagan.train(epochs=40000, batch_size=32)
