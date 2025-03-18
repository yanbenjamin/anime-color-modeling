#module for creating a custom Unet model, given some image dimensions
#and desired number of layers to perform image-to-image translation

import tensorflow as tf 
import tensorflow.keras 
import numpy as np

DROPOUT_RATE = 0.5
IMG_HEIGHT = 256
IMG_WIDTH = 256

def conv_block(num_filters,kernel_size = (3,3),apply_dropout = False, dropout_size = 0.4, strides = 1):
  tf_block = tf.keras.models.Sequential()
  tf_block.add(tf.keras.layers.Conv2D(num_filters, kernel_size, strides = strides, 
          activation = "relu", kernel_initializer = "he_normal", padding = "same"))
  
  if (apply_dropout == True): 
    tf_block.add(tf.keras.layers.Dropout(dropout_size))
  
  return tf_block 

def verify_EncoderDecoder_layers(encoder_layer_sizes, decoder_layer_sizes, 
                  encoder_dropouts, decoder_dropouts):
  assert len(encoder_layer_sizes) == len(encoder_dropouts) 
  assert len(decoder_layer_sizes) == len(decoder_dropouts) 
  assert len(encoder_layer_sizes) == len(decoder_layer_sizes) + 1
  assert len(decoder_layer_sizes) >= 1
  for idx in range(len(decoder_layer_sizes)): 
    assert decoder_layer_sizes[idx] == encoder_layer_sizes[-idx - 2]
    
class UNet_Builder(): #stride 2 units 
  def __init__(self, input_height = 256, input_width = 256, 
               kernel_size = (3,3),
               encoder_layer_sizes = [16,32,64,128,128,128],
               decoder_layer_sizes = [128,128,64,32,16], 
               encoder_dropouts = [False for _ in range(6)], 
               decoder_dropouts = [True,True,True,False,False],
               dropout_size = 0.4):
    
    #approval: ensuring that the layer sizes result in a functional model
    #and that layer sizes and dimensions in the UNet will work correctly. 
    verify_EncoderDecoder_layers(encoder_layer_sizes, decoder_layer_sizes, 
                  encoder_dropouts, decoder_dropouts)

    #input parameters
    self.input_height = input_height
    self.input_width = input_width
    self.grayscale_channels = 1 

    #setting parameters, after approval, as inputted by the user
    self.kernel_size = kernel_size 
    self.encoder_layer_sizes = encoder_layer_sizes 
    self.decoder_layer_sizes = decoder_layer_sizes 
    self.encoder_dropouts = encoder_dropouts 
    self.decoder_dropouts = decoder_dropouts 
    self.dropout_size = dropout_size

    #parameters used for upsampling and concatenation in the decoder stack
    self.down_strides = 2
    self.up_strides = 1
    self.upsampling_kernel_size = (2,2)

    #parameters for the final prediction layer 
    self.rgb_channels = 3
    self.rgb_kernel_size = (3,3)

  def set_image_dimensions(self, height, width):
    self.input_height = height 
    self.input_width = width 
  
  def set_kernel_size(self, kernel_size):
    self.kernel_size = kernel_size 
  
  def set_layers(self, encoder_layer_sizes, decoder_layer_sizes,
               encoder_dropouts, decoder_dropouts):
    verify_EncoderDecoder_layers(encoder_layer_sizes, decoder_layer_sizes, 
                  encoder_dropouts, decoder_dropouts)  
    self.encoder_layer_sizes = encoder_layer_sizes 
    self.decoder_layer_sizes = decoder_layer_sizes 
    self.encoder_dropouts = encoder_dropouts 
    self.decoder_dropouts = decoder_dropouts 

  def set_dropout_size(self,dropout_size): #a value in the [0,1] range
    self.dropout_size = dropout_size 
    
  def generate_model(self):

    input_layer = tf.keras.layers.Input(shape = (self.input_height, self.input_width, self.grayscale_channels))

    encoder_stack = []
    for layer_size, dropout in zip(self.encoder_layer_sizes, self.encoder_dropouts):
      encoder_stack.append(conv_block(layer_size, self.kernel_size, dropout, self.dropout_size, self.down_strides))
    
    decoder_stack = []
    for layer_size, dropout in zip(self.decoder_layer_sizes, self.decoder_dropouts):
      decoder_stack.append(conv_block(layer_size, self.kernel_size, dropout, self.dropout_size, self.up_strides))
    
    x = input_layer 
    intermediate_representations = []
    for encoder_block in encoder_stack: 
      x = encoder_block(x) 
      intermediate_representations.append(x)

    intermediate_representations = reversed(intermediate_representations[:-1])
    for decoder_block, intermediate in zip(decoder_stack, intermediate_representations):
      x = decoder_block(x)
      x = tf.keras.layers.UpSampling2D(self.upsampling_kernel_size)(x) 
      x = tf.keras.layers.Concatenate()([x, intermediate])
    
    prediction_layer = tf.keras.layers.Conv2DTranspose(
        self.rgb_channels, self.rgb_kernel_size, strides = self.down_strides, 
        padding = "same", kernel_initializer = "he_normal", activation = "tanh") 
    x = prediction_layer(x)

    colorization_model = tf.keras.Model(inputs = input_layer, outputs = x)
    return colorization_model

"""
main function primarily for debugging, testing, and peering into the Tensorflow model "blackbox"
"""
if __name__ == "__main__": 
  model_generator = UNet_Builder()
  deep_colorizing_model = model_generator.generate_model() #default image size is [256,256]

  #alters the model generator to create a more lightweight model for smaller images 
  model_generator.set_image_dimensions(128,128)
  model_generator.set_layers(encoder_layer_sizes = [16,32,64,128],
            decoder_layer_sizes = [64,32,16],
            encoder_dropouts = [False for _ in range(4)],
            decoder_dropouts = [True,True,False])
  lightweight_colorizing_model = model_generator.generate_model() 

  #testing the input and output dimensions of each model are correct
  for batch_size in [10,20,30,40,50]:
    noise_batch = np.random.rand(batch_size,256,256,1)
    assert deep_colorizing_model(noise_batch).shape == (batch_size,256,256,3)

    smaller_noise_batch = np.random.rand(batch_size, 128,128,1)
    assert lightweight_colorizing_model(smaller_noise_batch).shape == (batch_size,128,128,3)