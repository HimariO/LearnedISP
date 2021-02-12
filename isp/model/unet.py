import tensorflow as tf
from tensorflow.python.keras.engine.input_layer import Input

from . import base
from .io import dataset_element, model_prediction
from .layers import *
from isp import model


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
    return PySequential(layers=layers, name=name)
  
  def reverse_downsample_block(self, channel, name=None, WN=True):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=tf.nn.relu),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
    ]
    layers = [WeightNormalization(l, data_init=True, inference=not WN) for l in layers]
    return PySequential(layers=layers, name=name)
  
  def res_downsample_block(self, channel, name=None, WN=True):
    layers = [
      ResConvBlock(2, channel, weight_norm=WN),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=tf.nn.relu),
    ]
    layers = layers[:1] + [WeightNormalization(l, data_init=True, inference=not WN) for l in layers[1:]]
    return PySequential(layers=layers, name=name)
  
  def reverse_res_downsample_block(self, channel, name=None, WN=True):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=tf.nn.relu),
      ResConvBlock(2, channel, weight_norm=WN),
    ]
    layers = [WeightNormalization(l, data_init=True, inference=not WN) for l in layers[:-1]] + layers[-1:]
    return PySequential(layers=layers, name=name)
  
  def conv_block(self, channel, name=None, WN=True):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
    ]
    layers = [WeightNormalization(l, data_init=True, inference=not WN) for l in layers]
    return PySequential(layers=layers, name=name)
  
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
    return PySequential(layers=layers, name=name)

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
    return PySequential(layers=layers, name=name)
  
  def rgb_upsample_block(self, num_rgb_layer=0, name=None):
    layers = [
        tf.keras.layers.Conv2DTranspose(
            12, 3, padding='same', strides=(2, 2), activation=tf.nn.relu),
    ]
    layers += [tf.keras.layers.Conv2D(12, 3, padding='same', activation=tf.nn.relu) for _ in range(num_rgb_layer)]
    layers += [tf.keras.layers.Conv2D(3, 1)]
    return PySequential(layers=layers, name=name)


class UNetBilinearBlocks(UNetBlocks):

  def upsample_block(self, channel, name=None, WN=True):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
      tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
      tf.keras.layers.Conv2D(channel // 2, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
    ]
    
    layers = [
      WeightNormalization(l, data_init=True, inference=not WN)
      for l in layers[:-1]
    ] + [layers[-1]]
    return PySequential(layers=layers, name=name)

  def upsample_layer(self, channel, name=None, WN=True):
    layers = [
      tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
    ]
    return PySequential(layers)
  
  def res_upsample_block(self, channel, name=None, WN=True):
    layers = [
      ResConvBlock(2, channel, weight_norm=WN),
      tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
      tf.keras.layers.Conv2D(channel // 2, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
    ]
    return PySequential(layers=layers, name=name)
  
  def rgb_upsample_block(self, num_rgb_layer=1, name=None):
    layers = [
        tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
    ]
    layers += [tf.keras.layers.Conv2D(8, 3, padding='same', activation=tf.nn.relu) for _ in range(num_rgb_layer)]
    layers += [tf.keras.layers.Conv2D(3, 1)]
    return PySequential(layers=layers, name=name)


@base.register_model
class UNet(base.RawBase, UNetBlocks):

  def __init__(self, mode, *args, weight_decay_scale=0.00004, alpha=1.0, **kwargs):
    super().__init__(mode, *args, **kwargs)

    regularizer = tf.keras.regularizers.l2(weight_decay_scale)
    C = lambda channel: max(int(channel * alpha), 16)
    use_wn = True

    self.block_x1 = self.downsample_block(C(32), WN=use_wn)
    self.block_x2 = self.downsample_block(C(64), WN=use_wn)
    self.block_x4 = self.downsample_block(C(128), WN=use_wn)
    self.block_x8 = self.downsample_block(C(256), WN=use_wn)
    self.block_ux8 = self.upsample_block(C(512), WN=use_wn)
    self.block_ux4 = self.upsample_block(C(256), WN=use_wn)
    self.block_ux2 = self.upsample_block(C(128), WN=use_wn)
    self.block_ux1 = self.upsample_block(C(64), WN=use_wn)
    self.last_block = self.conv_block(C(32), WN=use_wn)
    self.transform = tf.keras.layers.Conv2D(12, 1, activation=None)
  
  def call(self, inputs, training=None, mask=None):
    raw = inputs[dataset_element.MAI_RAW_PATCH]
    rgb = self._call(raw)
    # import pdb; pdb.set_trace()
    return {
      model_prediction.ENHANCE_RGB: rgb
    }
  
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


@base.register_model
class UNetResX2R(base.RawBase, UNetBlocks):
  """
  R stand for reverse downsample block, which allow upsample block to concat features from deeper layers
  """

  def __init__(self, mode, *args, weight_decay_scale=0.00004,
               alpha=1.0, num_rgb_layer=0, weight_norm=True, **kwargs):
    super().__init__(mode, *args, **kwargs)

    regularizer = tf.keras.regularizers.l2(weight_decay_scale)

    def C(channel): return max(int(channel * alpha), 16)
    self.block_x1 = self.conv_block(C(32), WN=weight_norm)
    self.block_x2 = self.reverse_res_downsample_block(C(64), WN=weight_norm)
    self.block_x4 = self.reverse_res_downsample_block(C(128), WN=weight_norm)
    self.block_x8 = self.reverse_res_downsample_block(C(256), WN=weight_norm)
    self.block_x16 = self.reverse_res_downsample_block(C(512), WN=weight_norm)
    self.up_x16_x8 = self.upsample_layer(C(256))
    self.block_ux8 = self.res_conv_block(C(256), WN=weight_norm)
    self.up_x8_x4 = self.upsample_layer(C(128))
    self.block_ux4 = self.res_conv_block(C(128), WN=weight_norm)
    self.up_x4_x2 = self.upsample_layer(C(64))
    self.block_ux2 = self.res_conv_block(C(64), WN=weight_norm)
    self.up_x2_x1 = self.upsample_layer(C(32))
    self.last_conv = self.res_conv_block(C(32), WN=weight_norm)
    self.transform = self.rgb_upsample_block(num_rgb_layer=num_rgb_layer)

    self._first_kernel = None
  
  def call(self, inputs, training=None, mask=None):
    raw = inputs[dataset_element.MAI_RAW_PATCH]
    rgb = self._call(raw)
    # import pdb; pdb.set_trace()
    return {
      model_prediction.ENHANCE_RGB: rgb
    }

  def _call(self, x, training=None, mask=None):
    top = x
    x = x1 = self.block_x1(x)
    x = x2 = self.block_x2(x)
    x = x4 = self.block_x4(x)
    x = x8 = self.block_x8(x)
    x = self.block_x16(x)
    x = self.up_x16_x8(x)
    x = tf.concat([x, x8], axis=-1)
    x = self.block_ux8(x)
    x = self.up_x8_x4(x)
    x = tf.concat([x, x4], axis=-1)
    x = self.block_ux4(x)
    x = self.up_x4_x2(x)
    x = tf.concat([x, x2], axis=-1)
    x = self.block_ux2(x)
    x = self.up_x2_x1(x)
    x = tf.concat([x, x1], axis=-1)
    x = self.last_conv(x)
    x = self.transform(x)
    return x

  def swap_input_filter_order(self, flat_raw_pattern):
    assert type(flat_raw_pattern) is list
    assert all([0 <= i <= 3 for i in flat_raw_pattern])

    first_conv = self.block_x1.layers[0]
    if self._first_kernel is None:
      if isinstance(first_conv, tfa.layers.WeightNormalization):
        kernel = first_conv.v
      else:
        kernel = first_conv.kernel
      assert kernel.shape[2] == 4, f'{kernel.shape}'
      self._first_kernel = kernel

    filters = [self._first_kernel[:, :, i: i+1, :] for i in range(4)]
    new_kernel = [filters[color_code] for color_code in flat_raw_pattern]
    new_kernel = tf.concat(new_kernel, 2)
    if isinstance(first_conv, tfa.layers.WeightNormalization):
      first_conv.v = new_kernel
    else:
      first_conv.kernel = new_kernel


@base.register_model
class UNetRes(base.RawBase, UNetBilinearBlocks):
  """
  R stand for reverse downsample block, which allow upsample block to concat features from deeper layers
  """

  def __init__(self, mode, *args, weight_decay_scale=0.00004,
               alpha=1.0, num_rgb_layer=0, weight_norm=True, **kwargs):
    super().__init__(mode, *args, **kwargs)

    regularizer = tf.keras.regularizers.l2(weight_decay_scale)

    def C(channel): return max(int(channel * alpha), 16)
    self.block_x1 = self.conv_block(C(32), WN=weight_norm)
    self.block_x2 = self.reverse_res_downsample_block(C(64), WN=weight_norm)
    self.block_x4 = self.reverse_res_downsample_block(C(128), WN=weight_norm)
    self.block_x8 = self.reverse_res_downsample_block(C(256), WN=weight_norm)
    self.block_x16 = self.reverse_res_downsample_block(C(512), WN=weight_norm)
    self.up_x16_x8 = self.upsample_layer(C(256))
    self.block_ux8 = self.res_conv_block(C(256), WN=weight_norm)
    self.up_x8_x4 = self.upsample_layer(C(128))
    self.block_ux4 = self.res_conv_block(C(128), WN=weight_norm)
    self.up_x4_x2 = self.upsample_layer(C(64))
    self.block_ux2 = self.res_conv_block(C(64), WN=weight_norm)
    self.up_x2_x1 = self.upsample_layer(C(32))
    self.last_conv = self.res_conv_block(C(32), WN=weight_norm)
    # self.transform = tf.keras.layers.Conv2D(12, 1, activation=None)
    self.transform = self.rgb_upsample_block(num_rgb_layer=3)

    self._first_kernel = None
  
  def call(self, inputs, training=None, mask=None):
    raw = inputs[dataset_element.MAI_RAW_PATCH]
    rgb = self._call(raw)
    # import pdb; pdb.set_trace()
    return {
      model_prediction.ENHANCE_RGB: rgb
    }

  def _call(self, x, training=None, mask=None):
    top = x
    x = x1 = self.block_x1(x)
    x = x2 = self.block_x2(x)
    x = x4 = self.block_x4(x)
    x = x8 = self.block_x8(x)
    x = self.block_x16(x)
    x = self.up_x16_x8(x)
    x = tf.concat([x, x8], axis=-1)
    x = self.block_ux8(x)
    x = self.up_x8_x4(x)
    x = tf.concat([x, x4], axis=-1)
    x = self.block_ux4(x)
    x = self.up_x4_x2(x)
    x = tf.concat([x, x2], axis=-1)
    x = self.block_ux2(x)
    x = self.up_x2_x1(x)
    x = tf.concat([x, x1], axis=-1)
    x = self.last_conv(x)
    x = self.transform(x)
    
    # return tf.nn.depth_to_space(x, 2)
    return x

  def swap_input_filter_order(self, flat_raw_pattern):
    assert type(flat_raw_pattern) is list
    assert all([0 <= i <= 3 for i in flat_raw_pattern])

    first_conv = self.block_x1.layers[0]
    if self._first_kernel is None:
      if isinstance(first_conv, tfa.layers.WeightNormalization):
        kernel = first_conv.v
      else:
        kernel = first_conv.kernel
      assert kernel.shape[2] == 4, f'{kernel.shape}'
      self._first_kernel = kernel

    filters = [self._first_kernel[:, :, i: i+1, :] for i in range(4)]
    new_kernel = [filters[color_code] for color_code in flat_raw_pattern]
    new_kernel = tf.concat(new_kernel, 2)
    if isinstance(first_conv, tfa.layers.WeightNormalization):
      first_conv.v = new_kernel
    else:
      first_conv.kernel = new_kernel


@base.register_model
class UNetRes2Stage(base.RawBase, UNetBilinearBlocks):
  """
  R stand for reverse downsample block, which allow upsample block to concat features from deeper layers
  """

  def __init__(self, mode, *args, weight_decay_scale=0.00004,
               alpha=1.0, num_rgb_layer=0, weight_norm=True, **kwargs):
    super().__init__(mode, *args, **kwargs)

    regularizer = tf.keras.regularizers.l2(weight_decay_scale)

    def C(channel): return max(int(channel * alpha), 16)
    self.block_x1 = self.conv_block(C(32), WN=weight_norm)
    self.block_x2 = self.reverse_res_downsample_block(C(64), WN=weight_norm)
    self.block_x4 = self.reverse_res_downsample_block(C(128), WN=weight_norm)
    self.block_x8 = self.reverse_res_downsample_block(C(256), WN=weight_norm)
    self.block_x16 = self.reverse_res_downsample_block(C(512), WN=weight_norm)
    self.up_x16_x8 = self.upsample_layer(C(256))
    self.block_ux8 = self.res_conv_block(C(256), WN=weight_norm)
    self.up_x8_x4 = self.upsample_layer(C(128))
    self.block_ux4 = self.res_conv_block(C(128), WN=weight_norm)

    self.up_x4_x2 = self.upsample_layer(C(64))
    self.block_ux2 = self.res_conv_block(C(64), WN=weight_norm)
    self.up_x2_x1 = self.upsample_layer(C(32))
    self.last_conv = self.res_conv_block(C(32), WN=weight_norm)
    self.transform = tf.keras.layers.Conv2D(12, 1, activation=None)
    
    self.up_x4_x2_rec = self.upsample_layer(C(64))
    # self.block_ux2_rec = self.res_conv_block(C(64), WN=weight_norm)
    self.up_x2_x1_rec = self.upsample_layer(C(32))
    self.last_conv_rec = self.res_conv_block(C(32), WN=weight_norm)
    self.transform_rec = tf.keras.layers.Conv2D(4, 1, activation=None)

    self._first_kernel = None
  
  def call(self, inputs, training=None, mask=None):
    raw = inputs[dataset_element.MAI_RAW_PATCH]
    rgb, gray = self._call(raw)
    # import pdb; pdb.set_trace()
    return {
      model_prediction.ENHANCE_RGB: rgb,
      model_prediction.INTER_MID_GRAY: gray,
    }

  def _call(self, x, training=None, mask=None):
    top = x
    x = x1 = self.block_x1(x)
    x = x2 = self.block_x2(x)
    x = x4 = self.block_x4(x)
    x = x8 = self.block_x8(x)
    x = self.block_x16(x)
    x = self.up_x16_x8(x)
    x = tf.concat([x, x8], axis=-1)
    x = self.block_ux8(x)
    x = self.up_x8_x4(x)
    x = tf.concat([x, x4], axis=-1)
    x = ux4 = self.block_ux4(x)

    x = rec_x2 = self.up_x4_x2_rec(ux4)
    x = tf.concat([x, x2], axis=-1)
    # x = self.block_ux2_rec(x)
    x = rec_x1 = self.up_x2_x1_rec(x)
    x = tf.concat([x, x1], axis=-1)
    x = self.last_conv_rec(x)
    x_gray = tf.nn.depth_to_space(self.transform_rec(x), 2)
    
    x = self.up_x4_x2(ux4)
    x = tf.concat([x, x2, rec_x2], axis=-1)
    x = self.block_ux2(x)
    x = self.up_x2_x1(x)
    x = tf.concat([x, x1, rec_x1], axis=-1)
    x = self.last_conv(x)
    x_rgb = tf.nn.depth_to_space(self.transform(x), 2)

    return x_rgb, x_gray


@base.register_model
def functinoal_unet_res_2_stage(alpha=0.5, input_shape=[128, 128, 4]):
  unet = UNetRes2Stage('train', alpha=0.5)
  x_layer = tf.keras.Input(shape=input_shape)
  y1, y2 = unet._call(x_layer)
  y1 = tf.keras.layers.Lambda(lambda x: tf.identity(x), name=model_prediction.ENHANCE_RGB)(y1)
  y2 = tf.keras.layers.Lambda(lambda x: tf.identity(x), name=model_prediction.INTER_MID_GRAY)(y2)
  
  input_dict = {
    dataset_element.MAI_RAW_PATCH: x_layer
  }
  output_dict = {
    model_prediction.ENHANCE_RGB: y1,
    model_prediction.INTER_MID_GRAY: y2,
  }
  return tf.keras.Model(inputs=input_dict, outputs=output_dict)


@base.register_model
class UNetBay(base.RawBase, UNetBilinearBlocks):
  """
  R stand for reverse downsample block, which allow upsample block to concat features from deeper layers
  """

  def __init__(self, mode, *args, weight_decay_scale=0.00004,
               alpha=1.0, num_rgb_layer=0, weight_norm=True, **kwargs):
    super().__init__(mode, *args, **kwargs)

    regularizer = tf.keras.regularizers.l2(weight_decay_scale)

    def C(channel): return max(int(channel * alpha), 16)
    self.block_x1 = self.conv_block(C(32), WN=weight_norm)
    self.block_x2 = self.reverse_res_downsample_block(C(64), WN=weight_norm)
    self.block_x4 = self.reverse_res_downsample_block(C(128), WN=weight_norm)
    self.block_x8 = self.reverse_res_downsample_block(C(256), WN=weight_norm)
    self.block_x16 = self.reverse_res_downsample_block(C(512), WN=weight_norm)
    self.up_x16_x8 = self.upsample_layer(C(256))
    self.block_ux8 = self.res_conv_block(C(256), WN=weight_norm)
    self.up_x8_x4 = self.upsample_layer(C(128))
    self.block_ux4 = self.res_conv_block(C(128), WN=weight_norm)
    self.up_x4_x2 = self.upsample_layer(C(64))
    self.block_ux2 = self.res_conv_block(C(64), WN=weight_norm)
    self.up_x2_x1 = self.upsample_layer(C(32))
    self.last_conv = self.res_conv_block(C(32), WN=weight_norm)
    self.transform = tf.keras.layers.Conv2D(3, 1, activation=None)
    # self.transform = self.rgb_upsample_block(num_rgb_layer=3)

    self._first_kernel = None
  
  def call(self, inputs, training=None, mask=None):
    raw = inputs[dataset_element.MAI_RAW_PATCH]
    rgb = self._call(raw)
    # import pdb; pdb.set_trace()
    return {
      model_prediction.ENHANCE_RGB: rgb
    }
  
  def depthwise_bayer(self, x):
    r = x[..., 0:1]
    gr = x[..., 1:2]
    gb = x[..., 2:3]
    b = x[..., 3:4]

    p_r = tf.pad(r, [(0, 0), (0, 0), (0, 0), (0, 3)], constant_values=0.0)
    p_gr = tf.pad(r, [(0, 0), (0, 0), (0, 0), (1, 2)], constant_values=0.0)
    p_gb = tf.pad(r, [(0, 0), (0, 0), (0, 0), (2, 1)], constant_values=0.0)
    p_b = tf.pad(r, [(0, 0), (0, 0), (0, 0), (3, 0)], constant_values=0.0)

    dep_bayer = tf.concat([
      tf.nn.depth_to_space(p_r, 2),
      tf.nn.depth_to_space(p_gr, 2) + tf.nn.depth_to_space(p_gb, 2),
      tf.nn.depth_to_space(p_b, 2),
    ], axis=-1)
    return dep_bayer

  def _call(self, x, training=None, mask=None):
    x = self.depthwise_bayer(x)
    
    top = x
    x = x1 = self.block_x1(x)
    x = x2 = self.block_x2(x)
    x = x4 = self.block_x4(x)
    x = x8 = self.block_x8(x)
    x = self.block_x16(x)
    x = self.up_x16_x8(x)
    x = tf.concat([x, x8], axis=-1)
    x = self.block_ux8(x)
    x = self.up_x8_x4(x)
    x = tf.concat([x, x4], axis=-1)
    x = self.block_ux4(x)
    x = self.up_x4_x2(x)
    x = tf.concat([x, x2], axis=-1)
    x = self.block_ux2(x)
    x = self.up_x2_x1(x)
    x = tf.concat([x, x1], axis=-1)
    x = self.last_conv(x)
    x = self.transform(x)
    
    # return tf.nn.depth_to_space(x, 2)
    return x


@base.register_model
def functinoal_unet_bay(alpha=0.5, input_shape=[128, 128, 4]):
  unet = UNetBay('train', alpha=0.5)
  x_layer = tf.keras.Input(shape=input_shape)
  y1 = unet._call(x_layer)
  y1 = tf.keras.layers.Lambda(lambda x: tf.identity(x), name=model_prediction.ENHANCE_RGB)(y1)
  
  input_dict = {
    dataset_element.MAI_RAW_PATCH: x_layer
  }
  output_dict = {
    model_prediction.ENHANCE_RGB: y1,
  }
  model = tf.keras.Model(inputs=input_dict, outputs=output_dict)
  model.summary()
  return model