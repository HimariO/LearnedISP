import tensorflow as tf

from . import base
from .layers import *


class ResConvBlock(tf.keras.Model):

  def __init__(self, num_layer, channel, weight_norm=True, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._layers = []
    for i in range(num_layer):
      layer = tf.keras.layers.Conv2D(
          channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu)
      layer = WeightNormalization(
          layer, data_init=False, inference=(not weight_norm))
      # setattr(self, f'_conv_{i}', layer)
      self._layers.append(layer)

  def call(self, inputs):
    x = inputs
    for i, layer in enumerate(self._layers):
      t = x
      x = layer(x)
      if t.shape[-1] == x.shape[-1]:
        x += t
    return x


class UNetBlocks:

  def downsample_block(self, channel, name=None, WN=True):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=tf.nn.relu),
    ]
    layers = [WeightNormalization(l, data_init=True, inference=not WN) for l in layers]
    return tf.keras.Sequential(layers=layers, name=name)
  
  def reverse_downsample_block(self, channel, name=None, WN=True):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=tf.nn.relu),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
    ]
    layers = [WeightNormalization(l, data_init=True, inference=not WN) for l in layers]
    return tf.keras.Sequential(layers=layers, name=name)
  
  def res_downsample_block(self, channel, name=None, WN=True):
    layers = [
      ResConvBlock(2, channel, weight_norm=WN),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=tf.nn.relu),
    ]
    layers = layers[:1] + [WeightNormalization(l, data_init=True, inference=not WN) for l in layers[1:]]
    return tf.keras.Sequential(layers=layers, name=name)
  
  def reverse_res_downsample_block(self, channel, name=None, WN=True):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=tf.nn.relu),
      ResConvBlock(2, channel, weight_norm=WN),
    ]
    layers = [WeightNormalization(l, data_init=True, inference=not WN) for l in layers[:-1]] + layers[-1:]
    return tf.keras.Sequential(layers=layers, name=name)
  
  def conv_block(self, channel, name=None, WN=True):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
    ]
    layers = [WeightNormalization(l, data_init=True, inference=not WN) for l in layers]
    return tf.keras.Sequential(layers=layers, name=name)
  
  def res_conv_block(self, channel, name=None, WN=True):
    return ResConvBlock(2, channel, weight_norm=WN)
  
  def upsample_block(self, channel, name=None, WN=True):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
      tf.keras.layers.Conv2DTranspose(
        channel // 2, 3, padding='same', strides=(2, 2), activation=tf.nn.relu),
    ]
    layers = [WeightNormalization(l, data_init=True, inference=not WN) for l in layers[:-1]] + [layers[-1]]
    return tf.keras.Sequential(layers=layers, name=name)

  def upsample_layer(self, channel, name=None, WN=True):
    layer = tf.keras.layers.Conv2DTranspose(
      channel, 3, padding='same', strides=(2, 2), activation=tf.nn.relu)
    return layer
  
  def res_upsample_block(self, channel, name=None, WN=True):
    layers = [
      ResConvBlock(2, channel, weight_norm=WN),
      tf.keras.layers.Conv2DTranspose(
        channel // 2, 3, padding='same', strides=(2, 2), activation=tf.nn.relu),
    ]
    return tf.keras.Sequential(layers=layers, name=name)
  
  def rgb_upsample_block(self, num_rgb_layer=0, name=None):
    layers = [
        tf.keras.layers.Conv2DTranspose(
            12, 3, padding='same', strides=(2, 2), activation=tf.nn.relu),
    ]
    layers += [tf.keras.layers.Conv2D(12, 3, padding='same', activation=tf.nn.relu) for _ in range(num_rgb_layer)]
    layers += [tf.keras.layers.Conv2D(3, 1)]
    return tf.keras.Sequential(layers=layers, name=name)


@base.register_prediction_model
class UNet(base.RawBase, UNetBlocks):

  def __init__(self, mode, *args, weight_decay_scale=0.00004, alpha=1.0, **kwargs):
    super().__init__(mode, *args, **kwargs)

    regularizer = tf.keras.regularizers.l2(weight_decay_scale)
    C = lambda channel: max(int(channel * alpha), 16)

    self.block_x1 = self.downsample_block(C(32))
    self.block_x2 = self.downsample_block(C(64))
    self.block_x4 = self.downsample_block(C(128))
    self.block_x8 = self.downsample_block(C(256))
    self.block_ux8 = self.upsample_block(C(512))
    self.block_ux4 = self.upsample_block(C(256))
    self.block_ux2 = self.upsample_block(C(128))
    self.block_ux1 = self.upsample_block(C(64))
    self.last_block = self.conv_block(C(32))
    self.transform = tf.keras.layers.Conv2D(12, 1)
  
  def _call(self, x, training=None, mask=None):
    top = x
    x = x1 = self.block_x1(x)
    x = x2 = self.block_x2(x)
    x = x4 = self.block_x4(x)
    x = x8 = self.block_x8(x)
    x = self.block_ux8(x)
    x = tf.concat([x, x4], axis=-1)
    x = self.block_ux4(x)
    x = tf.concat([x, x2], axis=-1)
    x = self.block_ux2(x)
    x = tf.concat([x, x1], axis=-1)
    x = self.block_ux1(x)
    x = tf.concat([x, top], axis=-1)
    x = self.last_block(x)
    x = self.transform(x)
    return tf.nn.depth_to_space(x, 2)