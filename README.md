# GAN-keras
里面包含许多GAN算法的Keras源码，可以用于训练自己的模型。

# 仓库内容
- [x] gan
- [x] dcgan
- [x] cgan
- [x] acgan
- [x] cogan
- [x] cyclegan-keras
- [ ] cyclegan-pytorch

# 使用方法
## acgan、cgan、gan、dcgan、cogan
这些gan直接运行其中的代码。  
  
## cycleGAN
### 1、下载数据集
这是斑马to黄种马的数据集下载：  
[https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip)  
苹果to橘子数据集下载：  
[https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/apple2orange.zip]  (https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/apple2orange.zip)  
画作to照片数据集下载：  
[https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip)  
### 2、将数据集解压后放入到dataset中
### 3、运行cyclegan.py进行训练

# 参考
[https://github.com/eriklindernoren/Keras-GAN](https://github.com/eriklindernoren/Keras-GAN)  
[https://github.com/eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) 
