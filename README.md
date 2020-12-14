## GAN生成对抗网络在Keras当中的实现
---

## 注意事项
建立了一个新库可用于生成自己的图片文件，连接如下，具体gan种类为DCGAN：https://github.com/bubbliiiing/dcgan-keras  
执行过程可参考对应博客或者README.md，https://blog.csdn.net/weixin_44791964/article/details/110526044。

### 目录
1. [所需环境 Environment](#所需环境)
2. [仓库内容 WhatsIn](#仓库内容)
3. [使用方法 Usage](#使用方法)
4. [参考资料 Reference](#Reference)

### 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

### 仓库内容
- [x] gan
- [x] dcgan
- [x] cgan
- [x] acgan
- [x] cogan
- [x] srgan
- [x] cyclegan-keras
- [ ] cyclegan-pytorch

### 使用方法
#### acgan、cgan、gan、dcgan、cogan
这些gan直接运行其中的代码。  
  
#### cycleGAN
1、下载数据集  
这是斑马to黄种马的数据集下载：  
[https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip)  
苹果to橘子数据集下载：   
[https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/apple2orange.zip]  (https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/apple2orange.zip)   
画作to照片数据集下载：   
[https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip)   
2、将数据集解压后放入到dataset中  
3、运行cyclegan.py进行训练  

### Reference
[https://github.com/eriklindernoren/Keras-GAN](https://github.com/eriklindernoren/Keras-GAN)  
[https://github.com/eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) 
