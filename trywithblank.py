#coding=utf-8
from email import generator
import matplotlib
import tensorflow as tf
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np  
import os
from keras import backend as K
import random
from my_class import DataGenerator

filepath =r'./'
namelist=os.listdir(filepath + 'workflow/fulllabel')

def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2
def weightedLoss(w):

    def loss(true, pred):
        true=tf.cast(true, dtype=tf.float32)
        error = K.binary_crossentropy(true,pred,from_logits=True)
        error = K.switch(K.equal(true, 1), w * error , error)
        return error

    return loss

val, train = data_split(namelist, ratio=0.2, shuffle=True)

# Generators
batch_size=32
training_generator = DataGenerator(train)
validation_generator = DataGenerator(val)
#加载模型
#INIT_LR = 0.01
bce = tf.keras.losses.BinaryCrossentropy(reduction='sum')
EPOCHS =20
model = tf.keras.models.load_model('./MiSiDC04082020.h5',compile=False)
model.compile(optimizer='Adam',loss = weightedLoss(100),metrics=['accuracy'])
for ix, layer in enumerate(model.layers):
    if hasattr(model.layers[ix], 'kernel_initializer') and \
            hasattr(model.layers[ix], 'bias_initializer'):
        weight_initializer = model.layers[ix].kernel_initializer
        bias_initializer = model.layers[ix].bias_initializer

        old_weights, old_biases = model.layers[ix].get_weights()

        model.layers[ix].set_weights([
            weight_initializer(shape=old_weights.shape),
            bias_initializer(shape=len(old_biases))])

print("[INFO] training network...")     
H = model.fit(x=training_generator,validation_data=validation_generator,epochs=EPOCHS)

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('./training/20220519.png')

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save('./training/20220519.h5')
print('[INFO]finished!')