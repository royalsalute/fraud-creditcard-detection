from __future__ import print_function, division

import os

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Activation
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError, BinaryCrossentropy
from keras import backend as K
import tensorflow as tf
import numpy as np

from losses import encoder_loss, generator_loss, discriminator_loss, code_discriminator_loss


class AlphaGAN():
    def __init__(self, lambda_=1., lr1=0.0005, lr2=0.0001, beta1=0.9, beta2=0.999, model_save_path="./snapshots"):
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        self.model_save_path = model_save_path

        self.input_dim = 29
        self.x_shape = (self.input_dim, )
        self.latent_dim = 16
        self.base_n_count = 128

        self.lambda_ = lambda_
        self.lr1 = lr1
        self.lr2 = lr2
        self.beta1 = beta1
        self.beta2 = beta2

        self.bce = BinaryCrossentropy()
        self.mae = MeanAbsoluteError()

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.code_discriminator = self.build_code_discriminator()
        self.generator = self.build_generator()
        self.encoder = self.build_encoder()

        x = Input(shape=self.x_shape)
        x_hat = self.generator(self.encoder(x))
        self.alphagan_generator = Model([x], [x_hat])

    def build_encoder(self):
        model = Sequential(name="Encoder")

        model.add(Dense(self.base_n_count * 2))
        model.add(ReLU())
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.base_n_count))
        model.add(ReLU())
        model.add(BatchNormalization(momentum=0.8))
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
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.base_n_count * 2))
        model.add(ReLU())
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.base_n_count * 4))
        model.add(ReLU())
        model.add(BatchNormalization(momentum=0.8))
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
        model.add(Dropout(0.7))
        model.add(Dense(self.base_n_count * 2))
        model.add(LeakyReLU())
        model.add(Dropout(0.7))
        model.add(Dense(self.base_n_count))
        model.add(LeakyReLU())
        model.add(Dropout(0.7))
        model.add(Dense(1, activation='sigmoid'))

        x = Input(shape=self.x_shape)
        validity = model(x)
        model.summary()

        return Model(x, validity)

    def build_code_discriminator(self):
        model = Sequential(name="CodeDiscriminator")

        model.add(Dense(self.base_n_count * 4))
        model.add(LeakyReLU())
        model.add(Dropout(0.7))
        model.add(Dense(self.base_n_count * 2))
        model.add(LeakyReLU())
        model.add(Dropout(0.7))
        model.add(Dense(self.base_n_count))
        model.add(LeakyReLU())
        model.add(Dropout(0.7))
        model.add(Dense(1, activation='sigmoid'))

        z = Input(shape=(self.latent_dim,))
        validity = model(z)
        model.summary()

        return Model(z, validity)

    def build_e_train(self, batch_size, lr, beta1, beta2):
        x_real = K.placeholder(shape=(batch_size,) + self.x_shape)
        real_labels = K.placeholder(shape=(batch_size, 1))

        z_hat = self.encoder(x_real)
        c_z_hat = self.code_discriminator(z_hat)
        x_rec = self.generator(z_hat)

        # ================== Train E ================== #
        l1_loss = self.mae(x_real, x_rec)
        c_hat_loss = self.bce(c_z_hat, real_labels)  # - self.bce(c_z_hat, fake_labels)
        e_loss = l1_loss + c_hat_loss
        e_training_updates = Adam(lr=lr, beta_1=beta1, beta_2=beta2) \
            .get_updates(e_loss, self.encoder.trainable_weights)
        e_train = K.function([x_real, real_labels], [e_loss], updates=e_training_updates)

        return e_train

    def build_g_train(self, batch_size, lr, beta1, beta2):
        x_real = K.placeholder(shape=(batch_size,) + self.x_shape)
        z = K.placeholder(shape=(batch_size, self.latent_dim))
        real_labels = K.placeholder(shape=(batch_size, 1))
        fake_labels = K.placeholder(shape=(batch_size, 1))

        z_hat = self.encoder(x_real)

        x_rec = self.generator(z_hat)
        x_gen = self.generator(z)

        d_rec = self.discriminator(x_rec)
        d_gen = self.discriminator(x_gen)

        # ================== Train E ================== #
        l1_loss = 0.2 * self.mae(x_real, x_rec)
        g_rec_loss = self.bce(d_rec, real_labels)  # - self.bce(d_rec, fake_labels)
        g_gen_loss = self.bce(d_gen, fake_labels)  # - self.bce(d_gen, fake_labels)
        g_loss = l1_loss + g_rec_loss + g_gen_loss
        g_training_updates = Adam(lr=lr, beta_1=beta1, beta_2=beta2) \
            .get_updates(g_loss, self.generator.trainable_weights)
        g_train = K.function([x_real, z, real_labels, fake_labels], [g_loss], updates=g_training_updates)

        return g_train

    def build_e_g_train(self, batch_size, lr, beta1, beta2):
        x_real = K.placeholder(shape=(batch_size,) + self.x_shape)
        z = K.placeholder(shape=(batch_size, self.latent_dim))
        real_labels = K.placeholder(shape=(batch_size, 1))
        fake_labels = K.placeholder(shape=(batch_size, 1))

        z_hat = self.encoder(x_real)

        x_rec = self.generator(z_hat)
        x_gen = self.generator(z)

        d_rec = self.discriminator(x_rec)
        d_gen = self.discriminator(x_gen)

        c_z_hat = self.code_discriminator(z_hat)

        # ================== Train G and E ================== #
        l1_loss = self.mae(x_real, x_rec)
        c_hat_loss = self.bce(c_z_hat, real_labels)  # - self.bce(c_z_hat, fake_labels)
        g_rec_loss = self.bce(d_rec, real_labels)  # - self.bce(d_rec, fake_labels)
        g_gen_loss = self.bce(d_gen, real_labels)  # - self.bce(d_gen, fake_labels)
        g_loss = l1_loss + g_rec_loss + c_hat_loss + g_gen_loss
        g_training_updates = Adam(lr=lr, beta_1=beta1, beta_2=beta2) \
            .get_updates(g_loss, self.alphagan_generator.trainable_weights)
        g_train = K.function([x_real, z, real_labels, fake_labels], [g_loss], updates=g_training_updates)

        return g_train

    def build_d_train(self, batch_size, lr, beta1, beta2):
        x_real = K.placeholder(shape=(batch_size,) + self.x_shape)
        z = K.placeholder(shape=(batch_size, self.latent_dim))
        real_labels = K.placeholder(shape=(batch_size, 1))
        fake_labels = K.placeholder(shape=(batch_size, 1))

        z_hat = self.encoder(x_real)

        x_rec = self.generator(z_hat)
        x_gen = self.generator(z)

        d_real = self.discriminator(x_real)
        d_rec = self.discriminator(x_rec)
        d_gen = self.discriminator(x_gen)

        # ================== Train D ================== #
        d_real_loss = self.bce(d_real, real_labels)
        d_rec_loss = self.bce(d_rec, fake_labels)
        d_gen_loss = self.bce(d_gen, fake_labels)
        d_loss = d_real_loss + d_rec_loss + d_gen_loss
        d_training_updates = Adam(lr=lr, beta_1=beta1, beta_2=beta2) \
            .get_updates(d_loss, self.discriminator.trainable_weights)
        d_train = K.function([x_real, z, real_labels, fake_labels], [d_loss], updates=d_training_updates)

        return d_train

    def build_c_train(self, batch_size, lr, beta1, beta2):
        x_real = K.placeholder(shape=(batch_size,) + self.x_shape)
        z = K.placeholder(shape=(batch_size, self.latent_dim))
        real_labels = K.placeholder(shape=(batch_size, 1))
        fake_labels = K.placeholder(shape=(batch_size, 1))

        z_hat = self.encoder(x_real)
        c_z_hat = self.code_discriminator(z_hat)
        c_z = self.code_discriminator(z)

        # ================== Train C ================== #
        c_hat_loss = self.bce(c_z_hat, real_labels)
        c_z_loss = self.bce(c_z, fake_labels)
        c_loss = c_hat_loss + c_z_loss
        c_training_updates = Adam(lr=lr, beta_1=beta1, beta_2=beta2) \
            .get_updates(c_loss, self.code_discriminator.trainable_weights)
        c_train = K.function([x_real, z, real_labels, fake_labels], [c_loss], updates=c_training_updates)

        return c_train

    def train(self, X_train, epochs, batch_size=32, output_path='.', model_save_step=10):
        # if not os.path.exists(os.path.join(output_path, 'logs/')):
        #     os.makedirs(os.path.join(output_path, 'logs/'))

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # _, _, d_train, c_train = self.build_functions(batch_size, self.lr1, self.lr2, self.beta1, self.beta2)
        e_train = self.build_e_train(batch_size, lr=self.lr1, beta1=self.beta1, beta2=self.beta2)
        g_train = self.build_g_train(batch_size, lr=self.lr1, beta1=self.beta1, beta2=self.beta2)
        d_train = self.build_d_train(batch_size, lr=self.lr2, beta1=self.beta1, beta2=self.beta2)
        c_train = self.build_c_train(batch_size, lr=self.lr2, beta1=self.beta1, beta2=self.beta2)

        e_g_train = self.build_e_g_train(batch_size, lr=self.lr1, beta1=self.beta1, beta2=self.beta2)

        # train_step = self.build_train_step()

        # Adversarial ground truths

        session = K.get_session()
        init = tf.global_variables_initializer()
        session.run(init)

        for epoch in range(epochs):
            # Generate fake code
            z = np.random.normal(size=(batch_size, self.latent_dim)).astype(np.float32)
            # z_K.constant(z)

            # Make a batch of true samples
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            x_real = X_train[idx].astype(np.float32)

            # e_loss, g_loss, d_loss, c_loss, = train_step(x_real, z)

            #e_loss = e_train([x_real, real_labels])
            #g_loss = g_train([x_real, z, real_labels, fake_labels])
            g_loss = e_g_train([x_real, z, real_labels, fake_labels])
            d_loss = d_train([x_real, z, real_labels, fake_labels])
            c_loss = c_train([x_real, z, real_labels, fake_labels])
            # d_loss = d_train([x_real, z])
            # c_loss = c_train([x_real, z])

            # Plot the progress
            if epoch % 100 == 0:
                print("%d [E loss: %f] [G loss: %f] [D loss: %f] [C loss: %f]" % \
                      (epoch, 0, g_loss[0], d_loss[0], c_loss[0]))

            if epoch % model_save_step == 0:
                self.generator.save(os.path.join(self.model_save_path, '{}_G.h5'.format(epoch)))
                self.encoder.save(os.path.join(self.model_save_path, '{}_E.h5'.format(epoch)))
                self.discriminator.save(os.path.join(self.model_save_path, '{}_D.h5'.format(epoch)))
                self.code_discriminator.save(os.path.join(self.model_save_path, '{}_C.h5'.format(epoch)))

    def load_pretrained_models(self, model_path_prefix):
        self.generator.load_weights('%sG.h5' % model_path_prefix)
        self.encoder.load_weights('%sE.h5' % model_path_prefix)
        self.discriminator.load_weights('%sD.h5' % model_path_prefix)
        self.code_discriminator.load_weights('%sC.h5' % model_path_prefix)


# if __name__ == '__main__':
#     alphagan = AlphaGAN()
#     alphagan.train(epochs=40000, batch_size=32)



