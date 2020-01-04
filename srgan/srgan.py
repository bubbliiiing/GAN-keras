from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

import keras.backend as K

class SRGAN():
    def __init__(self):
        # 低分辨率图的shape
        self.channels = 3
        self.lr_height = 128
        self.lr_width = 128
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)


        # 高分辨率图的shape
        self.hr_height = self.lr_height*4
        self.hr_width = self.lr_width*4
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        
        # 16个残差卷积块
        self.n_residual_blocks = 16
        # 优化器
        optimizer = Adam(0.0002, 0.5)
        # 创建VGG模型，该模型用于提取特征
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        
        # 数据集
        self.dataset_name = 'DIV'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))


        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        # 建立判别模型
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.discriminator.summary()
        # 建立生成模型
        self.generator = self.build_generator()
        self.generator.summary()

        # 将生成模型和判别模型结合。训练生成模型的时候不训练判别模型。
        img_lr = Input(shape=self.lr_shape)

        fake_hr = self.generator(img_lr)
        fake_features = self.vgg(fake_hr)

        self.discriminator.trainable = False
        validity = self.discriminator(fake_hr)
        self.combined = Model(img_lr, [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[5e-1, 1],
                              optimizer=optimizer)


    def build_vgg(self):
        # 建立VGG模型，只使用第9层的特征
        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.hr_shape)
        img_features = vgg(img)

        return Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = BatchNormalization(momentum=0.8)(d)
            d = Activation('relu')(d)

            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u
        # 128,128,3
        img_lr = Input(shape=self.lr_shape)
        # 第一部分，低分辨率图像进入后会经过一个卷积+RELU函数
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        # 第二部分，经过16个残差网络结构，每个残差网络内部包含两个卷积+标准化+RELU，还有一个残差边。
        r = residual_block(c1, 64)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, 64)

        # 第三部分，上采样部分，将长宽进行放大，两次上采样后，变为原来的4倍，实现提高分辨率。
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        u1 = deconv2d(c2)
        u2 = deconv2d(u1)
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        # 由一堆的卷积+LeakyReLU+BatchNor构成
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, 64, bn=False)
        
        d2 = d_block(d1, 64, strides=2)
        d3 = d_block(d2, 128)
        d4 = d_block(d3, 128, strides=2)
        d5 = d_block(d4, 256)
        d6 = d_block(d5, 256, strides=2)
        d7 = d_block(d6, 512)
        d8 = d_block(d7, 512, strides=2)

        d9 = Dense(64*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def scheduler(self,models,epoch):
        # 学习率下降
        if epoch % 20000 == 0 and epoch != 0:
            for model in models:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))

    def train(self, epochs ,init_epoch=0, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()
        if init_epoch!= 0:
            self.generator.load_weights("weights/%s/gen_epoch%d.h5" % (self.dataset_name, init_epoch),skip_mismatch=True)
            self.discriminator.load_weights("weights/%s/dis_epoch%d.h5" % (self.dataset_name, init_epoch),skip_mismatch=True)

        for epoch in range(init_epoch,epochs):
            self.scheduler([self.combined,self.discriminator],epoch)
            # ---------------------- #
            #  训练判别网络
            # ---------------------- #
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            fake_hr = self.generator.predict(imgs_lr)

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------- #
            #  训练生成网络
            # ---------------------- #
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            valid = np.ones((batch_size,) + self.disc_patch)

            image_features = self.vgg.predict(imgs_hr)

            g_loss = self.combined.train_on_batch(imgs_lr, [valid, image_features])
            print(d_loss,g_loss)
            elapsed_time = datetime.datetime.now() - start_time
            print ("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, feature loss: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[1],
                                                                            g_loss[2],
                                                                            elapsed_time))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                # 保存
                if epoch % 500 == 0 and epoch != init_epoch:
                    os.makedirs('weights/%s' % self.dataset_name, exist_ok=True)
                    self.generator.save_weights("weights/%s/gen_epoch%d.h5" % (self.dataset_name, epoch))
                    self.discriminator.save_weights("weights/%s/dis_epoch%d.h5" % (self.dataset_name, epoch))

    def sample_images(self, epoch):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d.png" % (self.dataset_name, epoch))
        plt.close()

        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('images/%s/%d_lowres%d.png' % (self.dataset_name, epoch, i))
            plt.close()

if __name__ == '__main__':
    gan = SRGAN()
    gan.train(epochs=60000,init_epoch = 0, batch_size=1, sample_interval=50)
