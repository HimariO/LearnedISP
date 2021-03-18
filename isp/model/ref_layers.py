import functools

import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from isp.model.layers import PySequential

# BatchNormalization = functools.partial(
#   tf.keras.layers.Lambda,
#   lambda x: tf.contrib.slim.batch_norm(x, is_training=True))

class RepConv(tf.keras.Model):

  COUNT = 0

  def __init__(self,
              output_channel,
              filter_size,
              padding='same',
              strides=(1, 1),
              activation=tf.nn.relu6, 
              inference=False,
              norm_type='bn',
              is_training=True,
              **kwargs):
    super().__init__()
    self.channel = output_channel
    self.filter_size = filter_size
    self.padding = padding
    self.strides = strides
    self.activation = tf.keras.layers.Activation(activation)
    self.inference = inference
    self.norm_type = norm_type
    self.base_name = f'repconv_{RepConv.COUNT}'
    self.is_training = is_training
    self._build()
    RepConv.COUNT += 1
  
  def _build(self):
    self.conv3 = tf.keras.layers.Conv2D(
      self.channel, 3, strides=self.strides, padding=self.padding, use_bias=False, name=f'{self.name}_train_conv3')
    # self.conv3_1 = tf.keras.layers.Conv2D(
    #   self.channel, 1, strides=self.strides, padding=self.padding, use_bias=False)
    # self.conv3_2 = tf.keras.layers.Conv2D(
    #   self.channel, 1, strides=self.strides, padding=self.padding, use_bias=False)
    
    self.conv1 = tf.keras.layers.Conv2D(
      self.channel, 1, strides=self.strides, padding=self.padding, use_bias=False, name=f'{self.name}_train_conv1')
    self.add = tf.keras.layers.Add(name=f'{self.name}_add')
    
    if self.norm_type == 'bn':
      self.bn3 = BatchNormalization(name=f'{self.name}_bn_3')
      self.bn1 = BatchNormalization(name=f'{self.name}_bn_1')
      # self.bn_id = BatchNormalization(name=f'{self.name}_bn_id')
      self.bn_id = lambda x: x
    else:
      raise ValueError(f"Unknow normalization type: {self.norm_type}")

    self.rep_conv3 = tf.keras.layers.Conv2D(
      self.channel, 3, strides=self.strides, padding=self.padding, name=f'{self.name}_rep_conv3')
  
  def _fuse_bn_tensor(self, conv, bn):
    assert len(conv.variables) == 1, "conv layer of repvgg layer should't using bias!"
    kernel = conv.kernel  # (filter_size, filter_size, Ci, Co)
    gamma, beta, moving_mean, moving_var = bn.variables
    eps = bn.epsilon
    std = tf.math.sqrt(moving_var + eps)
    t = tf.reshape(gamma / std, [1, 1, -1, 1])
    return (kernel * t), (beta - moving_mean * gamma / std)
  
  def _fuse_bn_tensor_eager(self, conv, bn):
    assert len(conv.variables) == 1, "conv layer of repvgg layer should't using bias!"
    kernel = conv.kernel.numpy()
    gamma, beta, moving_mean, moving_var = [v.numpy() for v in bn.variables]
    eps = bn.epsilon.numpy()
  
  def reparameterization(self):
    if self.norm_type == 'bn':
      assert len(self.conv1.get_weights()) == 1
      assert len(self.conv3.get_weights()) == 1

      k3, b3 = self._fuse_bn_tensor(self.conv3, self.bn3)
      k1, b1 = self._fuse_bn_tensor(self.conv1, self.bn1)
      k1_3 = tf.pad(k1, [(1, 1), (1, 1), (0, 0), (0, 0)], constant_value=0.0)
      # if int(k1.shape[2]) == int(k1.shape[3]):
      #   kid = 

      raise NotImplementedError()
    else:
      raise ValueError(f"Unknow normalization type: {self.norm_type}")

  def call(self, inputs):
    return self._call(inputs)
  
  def _call(self, inputs):
    if self.inference:
      return self.rep_conv3(inputs)
    else:
      # x3 = self.bn3(self.conv3_2(self.conv3(self.conv3_1(inputs))))
      x3 = self.bn3(self.conv3(inputs))
      x1 = self.bn1(self.conv1(inputs))
      if int(inputs.shape[-1]) == self.channel:
        identity = self.bn_id(inputs)
        x = self.activation(self.add([x1, x3, identity]))
      else:
        x = self.activation(self.add([x1, x3]))
      return x
  
  def _shadow_call(self, i1, i2):
    """
    Used for quantization aware training to train with both reparameterized and
    non-reparameterized data path.
    NOTE: rep_conv should using shared kerenl obtain from _fuse_bn_tensor at this point
    """
    x3 = self.bn3(self.conv3(i1))
    x1 = self.bn1(self.conv1(i1))
    if int(i1.shape[-1]) == self.channel:
      identity = self.bn_id(i1)
      y1 = self.activation(self.add([x1, x3, identity]))
    else:
      y1 = self.activation(self.add([x1, x3]))
    
    y2 = self.rep_conv3(i2)
    return y1, y2


class RefRepVGGBlocks:

  def __init__(self, *args, **kwargs) -> None:
    RepConv.COUNT = 0  # NOTE: re-init layer counter so the layer count wont carry over to different model
    self.block_act = tf.nn.relu
    super().__init__(*args, **kwargs)
    mode = getattr(self, 'mode')
    self.is_training = mode == 'train' or mode == 'training' or mode == 'functional'
    logger.info(f"[{self.__class__.__name__}] is_training: {self.is_training}")

  def sequential(self, layers=None, name=None):
    if hasattr(self, 'mode'):
      if self.mode == 'export' or self.mode == 'functional':
        return PySequential(layers=layers, name=name)
      else:
        return tf.keras.Sequential(layers=layers, name=name)
    else:
      return tf.keras.Sequential(layers=layers, name=name)

  def downsample_block(self, channel, name=None, norm_type='wn'):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=None),
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type),
      RepConv(channel, 3, padding='same', strides=(2, 2), activation=self.block_act, norm_type=norm_type),
    ]
    return self.sequential(layers=layers, name=name)
  
  def reverse_downsample_block(self, channel, name=None, norm_type='wn'):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=None),
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type),
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type),
    ]
    return self.sequential(layers=layers, name=name)
  
  def res_downsample_block(self, channel, name=None, norm_type='wn'):
    layers = [
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type),
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=None),
    ]
    return self.sequential(layers=layers, name=name)
  
  def reverse_res_downsample_block(self, channel, name=None, norm_type='wn', num_block=2):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=None)] + [
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type)
      for _ in range(num_block)
    ]
    return self.sequential(layers=layers, name=name)
  
  def conv_block(self, channel, name=None, norm_type='wn', num_block=2):
    layers = [
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type)
      for _ in range(num_block)
    ]
    return self.sequential(layers=layers, name=name)
  
  def res_conv_block(self, channel, name=None, norm_type='wn', num_block=2):
    return self.conv_block(channel, name=name, norm_type=norm_type, num_block=num_block)
  
  def upsample_block(self, channel, name=None, norm_type='wn'):
    layers = [
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type),
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type),
      tf.keras.layers.Conv2DTranspose(
        channel // 2, 3, padding='same', strides=(2, 2), activation=None),
    ]
    return self.sequential(layers=layers, name=name)

  def upsample_layer(self, channel, name=None):
    layer = tf.keras.layers.Conv2DTranspose(
      channel, 3, padding='same', strides=(2, 2), activation=tf.nn.relu)
    return layer
  
  def res_upsample_block(self, channel, name=None, norm_type='wn'):
    layers = [
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type),
      tf.keras.layers.Conv2DTranspose(
        channel // 2, 3, padding='same', strides=(2, 2), activation=None),
    ]
    return self.sequential(layers=layers, name=name)
  
  def rgb_upsample_block(self, num_rgb_layer=0, name=None, norm_type='wn'):
    layers = [
      tf.keras.layers.Conv2DTranspose(
          12, 3, padding='same', strides=(2, 2), activation=None),
    ]
    layers += [
      RepConv(12, 3, padding='same', strides=(1, 1), activation=None, norm_type=norm_type)
      for _ in range(num_rgb_layer)
    ]
    layers += [tf.keras.layers.Conv2D(3, 1)]
    return self.sequential(layers=layers, name=name)


class RefRepBilinearVGGBlocks(RefRepVGGBlocks):

  def upsample_block(self, channel, name=None, norm_type='wn'):
    layers = [
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type),
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type),
      tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
      RepConv(channel // 2, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type),
    ]
    return self.sequential(layers=layers, name=name)

  def upsample_layer(self, channel, name=None, norm_type='wn'):
    layers = [
      tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type),
    ]
    return self.sequential(layers=layers, name=name)
  
  def res_upsample_block(self, channel, name=None, norm_type='wn'):
    layers = [
      RepConv(channel, 3, padding='same', strides=(1, 1), norm_type=norm_type, activation=self.block_act),
      RepConv(channel, 3, padding='same', strides=(1, 1), norm_type=norm_type, activation=self.block_act),
      tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
      RepConv(channel // 2, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type),
    ]
    return self.sequential(layers=layers, name=name)
  
  def rgb_upsample_block(self, num_rgb_layer=1, name=None, norm_type='wn'):
    layers = [
        tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
    ]
    layers += [
      RepConv(8, 3, padding='same', activation=None, norm_type=norm_type)
      for _ in range(num_rgb_layer)]
    layers += [tf.keras.layers.Conv2D(3, 1)]
    return self.sequential(layers=layers, name=name)