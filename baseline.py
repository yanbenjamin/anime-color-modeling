import tensorflow as tf
from tensorflow.keras import layers,losses
from tensorflow.keras.models import Sequential,Model
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob 

DROPOUT_RATE = 0.5
IMG_HEIGHT = 256
IMG_WIDTH = 256
INPUT_CHANNELS = 1 # just the L channel, or the grayscale
OUTPUT_CHANNELS = 3 #the R,G, B color channels are all generated with the baseline

def load_image_RGB(image_file):
  lab_img = cv2.imread(image_file) 
  if (lab_img.any() == None):
    raise ValueError(f"invalid image at {image_file}")
  
  #convert back to an RGB
  rgb_img = cv2.cvtColor(rgb_img, cv2.LAB2RGB)
  grayscale = lab_img[:,:,0] # the L channel for the luminosity
  return (grayscale - 127.5) / 127.5, (rgb_img - 127.5) / 127.5

def build_stack(img_files):

  train_stack_x = []
  train_stack_y = []
  for f in tqdm(img_files):
    L, AB = load_image_RGB(f)
    train_stack_x.append(np.expand_dims(L, -1))
    train_stack_y.append(AB)

  train_stack_x = np.array(train_stack_x)
  train_stack_y = np.array(train_stack_y)
  return train_stack_x, train_stack_y

train_stack_x, train_stack_y = build_stack(train_images_pipeline)
test_stack_x, test_stack_y = build_stack(test_images_pipeline)
print(train_stack_x.shape, train_stack_y.shape)
print(test_stack_x.shape, test_stack_y.shape)

def conv_block(filters, kernel, apply_dropout=False, strides=1, apply_batchnorm=True):
    block = tf.keras.models.Sequential()
    block.add(tf.keras.layers.Conv2D(filters,kernel,strides=strides,kernel_initializer="he_normal",
                                      padding="same"))
    if (apply_batchnorm):
        block.add(tf.keras.layers.BatchNormalization())
    
    if (apply_dropout == True):
        block.add(tf.keras.layers.Dropout(DROPOUT_RATE))
        block.add(tf.keras.layers.ReLU())
    
    return block

def create_UNet():
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 1])
    KERNEL_SIZE = (3,3) 
    UP_KERNEL_SIZE = (2,2) #kernel used for 2D upsampling at second part of UNet model

    encoder_stack = [
        conv_block(IMG_HEIGHT // 8,KERNEL_SIZE,False,2,False),
        conv_block(IMG_HEIGHT // 4,KERNEL_SIZE,False, strides = 2),
        conv_block(IMG_HEIGHT // 2,KERNEL_SIZE,False, strides = 2),
        conv_block(IMG_HEIGHT // 2,KERNEL_SIZE,False, strides = 2),
        conv_block(IMG_HEIGHT,KERNEL_SIZE,False, strides = 2),
    ]
    
    decoder_stack = [
        conv_block(IMG_HEIGHT,KERNEL_SIZE,True, strides = 1),
        conv_block(IMG_HEIGHT,KERNEL_SIZE,True, strides = 1),
        conv_block(IMG_HEIGHT // 2,KERNEL_SIZE,False, strides = 1),
        conv_block(IMG_HEIGHT // 2,KERNEL_SIZE,False, strides = 1),
        conv_block(IMG_HEIGHT // 4,KERNEL_SIZE,False,1),
        conv_block(IMG_HEIGHT // 8,KERNEL_SIZE,False,1),
    ]

    x = inputs

    intermediates = []
    for block in encoder_stack:
        x = block(x)
        intermediates.append(x)

    intermediates = reversed(intermediates[:-1])

    for block,inter in zip(decoder_stack,intermediates):
        x = block(x)
        x = tf.keras.layers.UpSampling2D(UP_KERNEL_SIZE)(x)
        x = tf.keras.layers.Concatenate()([x,inter])

    final_layer = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, KERNEL_SIZE,strides=2,padding='same',kernel_initializer="he_normal",activation='tanh')
    x = final_layer(x)

    return tf.keras.Model(inputs=inputs,outputs=x)

if __name__ == "__main__":

    EPOCHS = 100

    #build and train the model on the dataset 
    model = create_UNet()
    model.summary()

    #read from the files that preprocessing just created 
    train_images_pipeline = glob.glob("./train/*")
    test_images_pipeline = glob.glob("./test/*")

    train_stack_x, train_stack_y = build_stack(train_images_pipeline)
    test_stack_x, test_stack_y = build_stack(test_images_pipeline)
    print("Train Data", train_stack_x.shape, train_stack_y.shape)
    print("Test Data", test_stack_x.shape, test_stack_y.shape)

    #compile and train the model
    model.compile(loss='mean_squared_error', optimizer='adam') #, metrics=['mean_absolute_error'])

    logs = model.fit(train_stack_x, train_stack_y,
                  epochs=EPOCHS, validation_data = (test_stack_x, test_stack_y)) #,
    
    #save out the model for later inference
    fname_to_save_to = "manga-anime-model-baseline.keras"
    model.save(fname_to_save_to)