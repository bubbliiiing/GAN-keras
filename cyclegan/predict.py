from nets.resnet import get_resnet
from PIL import Image
import numpy as np
model = get_resnet(None,None,3)
model.load_weights(r"weights\horse2zebra\g_BA_epoch15.h5")

img = np.array(Image.open(r"datasets\horse2zebra\trainB\n02391049_32.jpg").resize([256,256]))/127.5 - 1
img = np.expand_dims(img,axis=0)
fake = (model.predict(img)*0.5 + 0.5)*255

face = Image.fromarray(np.uint8(fake[0]))
face.show()