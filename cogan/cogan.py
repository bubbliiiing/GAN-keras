from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
import numpy as np

class COGAN():
    def __init__(self):
        # 输入shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # 分十类
        self.num_classes = 10
        self.latent_dim = 100
        # adam优化器
        optimizer = Adam(0.0002, 0.5)
        # 生成两个判别器
        self.d1, self.d2 = self.build_discriminators()
        self.d1.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d2.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        # 建立生成器
        self.g1, self.g2 = self.build_generators()

        z = Input(shape=(self.latent_dim,))
        img1 = self.g1(z)
        img2 = self.g2(z)

        self.d1.trainable = False
        self.d2.trainable = False

        valid1 = self.d1(img1)
        valid2 = self.d2(img2)

        self.combined = Model(z, [valid1, valid2])
        self.combined.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                                    optimizer=optimizer)

    def build_generators(self):
        # 共享权值部分
        noise = Input(shape=(self.latent_dim,))
        x = Dense(32 * 7 * 7, activation="relu", input_dim=self.latent_dim)(noise)
        x = Reshape((7, 7, 32))(x)
        
        x = Conv2D(64, kernel_size=3, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("relu")(x)

        x = UpSampling2D()(x)
        x = Conv2D(128, kernel_size=3, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("relu")(x)

        x = UpSampling2D()(x)
        x = Conv2D(128, kernel_size=3, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        feature_repr = Activation("relu")(x)
        model = Model(noise,feature_repr)

        noise = Input(shape=(self.latent_dim,))
        feature_repr = model(noise)
        # 生成模型1
        g1 = Conv2D(64, kernel_size=1, padding="same")(feature_repr)
        g1 = BatchNormalization(momentum=0.8)(g1)
        g1 = Activation("relu")(g1)

        g1 = Conv2D(64, kernel_size=3, padding="same")(g1)
        g1 = BatchNormalization(momentum=0.8)(g1)
        g1 = Activation("relu")(g1)

        g1 = Conv2D(64, kernel_size=1, padding="same")(g1)
        g1 = BatchNormalization(momentum=0.8)(g1)
        g1 = Activation("relu")(g1)

        g1 = Conv2D(self.channels, kernel_size=1, padding="same")(g1)
        img1 = Activation("tanh")(g1)

        # 生成模型2
        g2 = Conv2D(64, kernel_size=1, padding="same")(feature_repr)
        g2 = BatchNormalization(momentum=0.8)(g2)
        g2 = Activation("relu")(g2)
        
        g2 = Conv2D(64, kernel_size=3, padding="same")(g2)
        g2 = BatchNormalization(momentum=0.8)(g2)
        g2 = Activation("relu")(g2)

        g2 = Conv2D(64, kernel_size=1, padding="same")(g2)
        g2 = BatchNormalization(momentum=0.8)(g2)
        g2 = Activation("relu")(g2)

        g2 = Conv2D(self.channels, kernel_size=1, padding="same")(g2)
        img2 = Activation("tanh")(g2)

        return Model(noise, img1), Model(noise, img2)


    def build_discriminators(self):

        # 共享权值部分
        img = Input(shape=self.img_shape)
        x = Conv2D(64, kernel_size=3, strides=2, padding="same")(img)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("relu")(x)

        x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("relu")(x)

        x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = GlobalAveragePooling2D()(x)
        feature_repr = Activation("relu")(x)

        model = Model(img,feature_repr)

        img1 = Input(shape=self.img_shape)
        img2 = Input(shape=self.img_shape)
        img1_embedding = model(img1)
        img2_embedding = model(img2)
        # 生成评价模型1
        validity1 = Dense(1, activation='sigmoid')(img1_embedding)
        # 生成评价模型2
        validity2 = Dense(1, activation='sigmoid')(img2_embedding)

        return Model(img1, validity1), Model(img2, validity2)

    def train(self, epochs, batch_size=128, sample_interval=50):
        (X_train, _), (_, _) = mnist.load_data()

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        X1 = X_train[:int(X_train.shape[0]/2)]
        X2 = X_train[int(X_train.shape[0]/2):]
        X2 = scipy.ndimage.interpolation.rotate(X2, 90, axes=(1, 2))

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------- #
            #  训练评价者
            # ---------------------- #

            idx = np.random.randint(0, X1.shape[0], batch_size)
            imgs1 = X1[idx]
            imgs2 = X2[idx]

            noise = np.random.normal(0, 1, (batch_size, 100))

            gen_imgs1 = self.g1.predict(noise)
            gen_imgs2 = self.g2.predict(noise)

            d1_loss_real = self.d1.train_on_batch(imgs1, valid)
            d2_loss_real = self.d2.train_on_batch(imgs2, valid)
            d1_loss_fake = self.d1.train_on_batch(gen_imgs1, fake)
            d2_loss_fake = self.d2.train_on_batch(gen_imgs2, fake)
            d1_loss = 0.5 * np.add(d1_loss_real, d1_loss_fake)
            d2_loss = 0.5 * np.add(d2_loss_real, d2_loss_fake)


            # ------------------ #
            #  训练生成模型
            # ------------------ #

            g_loss = self.combined.train_on_batch(noise, [valid, valid])

            print ("%d [D1 loss: %f, acc.: %.2f%%] [D2 loss: %f, acc.: %.2f%%] [G loss: %f]" \
                % (epoch, d1_loss[0], 100*d1_loss[1], d2_loss[0], 100*d2_loss[1], g_loss[0]))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * int(c/2), 100))
        gen_imgs1 = self.g1.predict(noise)
        gen_imgs2 = self.g2.predict(noise)
        gen_imgs = np.concatenate([gen_imgs1, gen_imgs2])

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    if not os.path.exists("./images"):
        os.makedirs("./images")
    gan = COGAN()
    gan.train(epochs=30000, batch_size=256, sample_interval=200)
