import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.engine.input_layer import Input

from . import base
from .io import dataset_element, model_prediction
from .layers import *
from isp import model


@base.register_model
class UNet(base.RawBase, UNetBilinearBlocks):

  def __init__(self, mode, *args, weight_decay_scale=0.00004, alpha=1.0, **kwargs):
    super().__init__(mode, *args, **kwargs)

    regularizer = tf.keras.regularizers.l2(weight_decay_scale)
    C = lambda channel: max(int(channel * alpha), 16)
    use_wn = False

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
    x = tf.keras.layers.Concatenate(axis=-1)([x, x4])
    x = self.block_ux4(x)
    x = tf.keras.layers.Concatenate(axis=-1)([x, x2])
    x = self.block_ux2(x)
    x = tf.keras.layers.Concatenate(axis=-1)([x, x1])
    x = self.block_ux1(x)
    x = tf.keras.layers.Concatenate(axis=-1)([x, top])
    x = self.last_block(x)
    x = self.transform(x)
    x = tf.keras.layers.Lambda(lambda z: tf.nn.depth_to_space(z, 2))(x)
    return x


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
    self.transform = tf.keras.layers.Conv2D(12, 1, activation=None)
    # self.transform = self.rgb_upsample_block(num_rgb_layer=3)

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
    
    return tf.nn.depth_to_space(x, 2)
    # return x

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
    
    self.up_x4_x2_rec = self.upsample_layer(C(32))
    # self.block_ux2_rec = self.res_conv_block(C(64), WN=weight_norm)
    self.up_x2_x1_rec = self.upsample_layer(8)
    self.last_conv_rec = self.res_conv_block(8, WN=weight_norm)
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
  y2 = tf.keras.layers.Lambda(lambda x: tf.identity(x), name=model_prediction.INTER_MID_PRED)(y2)
  
  input_dict = {
    dataset_element.MAI_RAW_PATCH: x_layer
  }
  output_dict = {
    model_prediction.ENHANCE_RGB: y1,
    model_prediction.INTER_MID_PRED: y2,
  }
  model = tf.keras.Model(inputs=input_dict, outputs=output_dict)
  model.summary()
  return model


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


@base.register_model
class UNetCURL(base.RawBase, RepBilinearVGGBlocks):
  """
  R stand for reverse downsample block, which allow upsample block to concat features from deeper layers
  """

  def __init__(self, mode, *args, weight_decay_scale=0.00004,
               alpha=1.0, num_rgb_layer=0, norm_type='bn', **kwargs):
    super().__init__(mode, *args, **kwargs)

    regularizer = tf.keras.regularizers.l2(weight_decay_scale)

    def C(channel): return max(int(channel * alpha), 16)
    self.block_x1 = self.conv_block(C(32), norm_type=norm_type)
    self.block_x2 = self.reverse_res_downsample_block(C(64), norm_type=norm_type)
    self.block_x4 = self.reverse_res_downsample_block(C(128), norm_type=norm_type)
    self.block_x8 = self.reverse_res_downsample_block(C(256), norm_type=norm_type)
    self.block_x16 = self.reverse_res_downsample_block(C(512), norm_type=norm_type)
    self.up_x16_x8 = self.upsample_layer(C(256), norm_type=norm_type)
    self.block_ux8 = self.res_conv_block(C(256), norm_type=norm_type)
    self.up_x8_x4 = self.upsample_layer(C(128), norm_type=norm_type)
    self.block_ux4 = self.res_conv_block(C(128), norm_type=norm_type)
    self.up_x4_x2 = self.upsample_layer(C(64), norm_type=norm_type)
    self.block_ux2 = self.res_conv_block(C(64), norm_type=norm_type)
    self.up_x2_x1 = self.upsample_layer(C(32), norm_type=norm_type)
    self.last_conv = self.res_conv_block(C(32), norm_type=norm_type)
    self.transform = tf.keras.layers.Conv2D(12, 1, activation=None)
    # self.transform = self.rgb_upsample_block(num_rgb_layer=3)

    self.curl_raw_ds = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
    self.curl_rgb_ds = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))
    self.curl_x1 = self.reverse_res_downsample_block(C(64), norm_type=norm_type)
    self.curl_x2 = self.reverse_res_downsample_block(C(128), norm_type=norm_type)
    self.curl_gap = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')
    self.curl_poly_fc = tf.keras.layers.Dense(16 * 3)
    self.curl_poly_r = PLCurve()
    self.curl_poly_g = PLCurve()
    self.curl_poly_b = PLCurve()

    self._first_kernel = None
  
  def call(self, inputs, training=None, mask=None):
    raw = inputs[dataset_element.MAI_RAW_PATCH]
    rgb, adj_rgb, rgb_curl = self._call(raw)
    # import pdb; pdb.set_trace()
    return {
      model_prediction.ENHANCE_RGB: adj_rgb,
      model_prediction.INTER_MID_PRED: rgb,
      # model_prediction.RGB_CURL: rgb_curl,
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
    x = up_x2 = tf.concat([x, x2], axis=-1)
    x = self.block_ux2(x)
    x = up_x1 = self.up_x2_x1(x)
    x = tf.concat([x, x1], axis=-1)
    x = self.last_conv(x)
    x = self.transform(x)
    
    rgb = tf.nn.depth_to_space(x, 2)
    r, g, b = tf.split(rgb, 3, axis=-1)
    
    ds_raw = self.curl_raw_ds(top)
    ds_rgb = self.curl_rgb_ds(rgb)
    x_curl = self.curl_x1(tf.concat([ds_raw, ds_rgb], -1))
    x_curl = self.curl_x2(x_curl)
    x_curl = self.curl_gap(x_curl)
    x_curl = self.curl_poly_fc(x_curl)
    
    r = self.curl_poly_r(tf.stop_gradient(r), x_curl[:,:16])
    g = self.curl_poly_g(tf.stop_gradient(g), x_curl[:, 16:32])
    b = self.curl_poly_b(tf.stop_gradient(b), x_curl[:, 32:])
    # import pdb; pdb.set_trace()
    adj_rgb = tf.concat([r, g, b], axis=-1)
    return rgb, adj_rgb, x_curl
  
  def _call_hsv(self, x, training=None, mask=None):
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
    x = up_x2 = tf.concat([x, x2], axis=-1)
    x = self.block_ux2(x)
    x = up_x1 = self.up_x2_x1(x)
    x = tf.concat([x, x1], axis=-1)
    x = self.last_conv(x)
    x = self.transform(x)
    
    rgb = tf.nn.depth_to_space(x, 2)
    hsv = tf.image.rgb_to_hsv(rgb)
    h, s, v = tf.split(hsv, 3, axis=-1)
    
    ds_raw = self.curl_raw_ds(top)
    ds_rgb = self.curl_rgb_ds(rgb)
    x_curl = self.curl_x1(tf.concat([ds_raw, ds_rgb], -1))
    x_curl = self.curl_x2(x_curl)
    x_curl = self.curl_gap(x_curl)
    x_curl = self.curl_poly_fc(x_curl)
    
    h = self.curl_poly_r(tf.stop_gradient(h), x_curl[:,:16])
    s = self.curl_poly_g(tf.stop_gradient(s), x_curl[:, 16:32])
    v = self.curl_poly_b(tf.stop_gradient(v), x_curl[:, 32:])
    # import pdb; pdb.set_trace()
    adj_hsv = tf.concat([h, s, v], axis=-1)
    adj_rgb = tf.image.hsv_to_rgb(adj_hsv)
    return rgb, adj_rgb, x_curl


@base.register_model
class UNetHSV(base.RawBase, UNetBilinearBlocks):
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
    self.to_hsv = tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))
    # self.transform = self.rgb_upsample_block(num_rgb_layer=3)

    self._first_kernel = None
  
  def call(self, inputs, training=None, mask=None):
    raw = inputs[dataset_element.MAI_RAW_PATCH]
    rgb, hsv = self._call(raw)
    # import pdb; pdb.set_trace()
    return {
      model_prediction.ENHANCE_RGB: rgb,
      model_prediction.INTER_MID_PRED: hsv,
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
    
    rgb = tf.nn.depth_to_space(x, 2)
    hsv = self.to_hsv(rgb)
    return rgb, hsv
    # return x


@base.register_model
class UNetRes3Stage(base.RawBase, UNetBilinearBlocks):
  """
  Graysclae --> RGB --> Color curve adjusted RGB
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
    
    self.up_x4_x2_rec = self.upsample_layer(C(32))
    # self.block_ux2_rec = self.res_conv_block(C(64), WN=weight_norm)
    self.up_x2_x1_rec = self.upsample_layer(8)
    self.last_conv_rec = self.res_conv_block(8, WN=weight_norm)
    self.transform_rec = tf.keras.layers.Conv2D(4, 1, activation=None)

    self.curl_x1 = self.reverse_res_downsample_block(C(64), WN=weight_norm)
    self.curl_x2 = self.reverse_res_downsample_block(C(128), WN=weight_norm)
    self.curl_gap = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')
    self.curl_poly_fc = tf.keras.layers.Dense(16 * 3)
    self.curl_poly_r = PLCurve()
    self.curl_poly_g = PLCurve()
    self.curl_poly_b = PLCurve()

    self._first_kernel = None
  
  def call(self, inputs, training=None, mask=None):
    raw = inputs[dataset_element.MAI_RAW_PATCH]
    adj_rgb, rgb, gray = self._call(raw)
    # import pdb; pdb.set_trace()
    return {
      model_prediction.ENHANCE_RGB: adj_rgb,
      model_prediction.INTER_MID_PRED: rgb,
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
    x = up_x1 = self.up_x2_x1(x)
    x = tf.concat([x, x1, rec_x1], axis=-1)
    x = self.last_conv(x)
    x_rgb = tf.nn.depth_to_space(self.transform(x), 2)

    r, g, b = tf.split(x_rgb, 3, axis=-1)
    
    x_curl = self.curl_x1(up_x1)
    x_curl = self.curl_x2(x_curl)
    x_curl = self.curl_gap(x_curl)
    x_curl = self.curl_poly_fc(x_curl)
    
    r = self.curl_poly_r(tf.stop_gradient(r), x_curl[:,:16])
    g = self.curl_poly_g(tf.stop_gradient(g), x_curl[:, 16:32])
    b = self.curl_poly_b(tf.stop_gradient(b), x_curl[:, 32:])
    # import pdb; pdb.set_trace()
    adj_rgb = tf.concat([r, g, b], axis=-1)

    return adj_rgb, x_rgb, x_gray


@base.register_model
class UNetGrid(base.RawBase, UNetBilinearBlocks):
  """
  R stand for reverse downsample block, which allow upsample block to concat features from deeper layers
  """

  def __init__(self, mode, *args, weight_decay_scale=0.00004,
               alpha=1.0, num_rgb_layer=0, norm_type='bn', **kwargs):
    super().__init__(mode, *args, **kwargs)

    regularizer = tf.keras.regularizers.l2(weight_decay_scale)

    def C(channel): return max(int(channel * alpha), 16)
    self.coord = ConcatCoordinate()
    self.block_x1 = self.conv_block(C(32), norm_type=norm_type)
    self.block_x2 = self.reverse_res_downsample_block(C(64), norm_type=norm_type)
    self.block_x4 = self.reverse_res_downsample_block(C(128), norm_type=norm_type)
    self.block_x8 = self.reverse_res_downsample_block(C(256), norm_type=norm_type)
    self.block_x16 = self.reverse_res_downsample_block(C(512), norm_type=norm_type)
    self.up_x16_x8 = self.upsample_layer(C(256), norm_type=norm_type)
    self.block_ux8 = self.res_conv_block(C(256), norm_type=norm_type)
    self.up_x8_x4 = self.upsample_layer(C(128), norm_type=norm_type)
    self.block_ux4 = self.res_conv_block(C(128), norm_type=norm_type)
    self.up_x4_x2 = self.upsample_layer(C(64), norm_type=norm_type)
    self.block_ux2 = self.res_conv_block(C(64), norm_type=norm_type)
    self.up_x2_x1 = self.upsample_layer(C(32), norm_type=norm_type)
    self.last_conv = self.res_conv_block(C(32), norm_type=norm_type)
    # self.transform = tf.keras.layers.Conv2D(12, 1, activation=None)
    self.transform = self.rgb_upsample_block(num_rgb_layer=1, norm_type=norm_type)

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
    # x = self.coord(x)
    x = x1 = self.coord(self.block_x1(x))
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
class DeepUNetGrid(base.RawBase, RepBilinearVGGBlocks):
  """
  R stand for reverse downsample block, which allow upsample block to concat features from deeper layers
  """

  def __init__(self, mode, *args, weight_decay_scale=0.00004,
               alpha=1.0, num_rgb_layer=0, norm_type='bn', **kwargs):
    super().__init__(mode, *args, **kwargs)

    regularizer = tf.keras.regularizers.l2(weight_decay_scale)

    def C(channel): return max(int(channel * alpha), 16)
    self.coord = ConcatCoordinate()
    self.block_x1 = self.conv_block(C(32), norm_type=norm_type)
    self.block_x2 = self.reverse_res_downsample_block(C(64), norm_type=norm_type, num_block=4)
    self.block_x4 = self.reverse_res_downsample_block(C(128), norm_type=norm_type, num_block=4)
    self.block_x8 = self.reverse_res_downsample_block(C(256), norm_type=norm_type, num_block=4)
    self.block_x16 = self.reverse_res_downsample_block(C(512), norm_type=norm_type, num_block=4)
    self.up_x16_x8 = self.upsample_layer(C(256), norm_type=norm_type)
    self.block_ux8 = self.res_conv_block(C(256), norm_type=norm_type, num_block=4)
    self.up_x8_x4 = self.upsample_layer(C(128), norm_type=norm_type)
    self.block_ux4 = self.res_conv_block(C(128), norm_type=norm_type, num_block=4)
    self.up_x4_x2 = self.upsample_layer(C(64), norm_type=norm_type)
    self.block_ux2 = self.res_conv_block(C(64), norm_type=norm_type, num_block=4)
    self.up_x2_x1 = self.upsample_layer(C(32), norm_type=norm_type)
    self.last_conv = self.res_conv_block(C(32), norm_type=norm_type, num_block=4)
    # self.transform = tf.keras.layers.Conv2D(12, 1, activation=None)
    self.transform = self.rgb_upsample_block(num_rgb_layer=1, norm_type=norm_type)

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
    # x = self.coord(x)
    x = x1 = self.coord(self.block_x1(x))
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
class UNetCoBi(UNetGrid):

  def __init__(self, mode, *args, weight_decay_scale=0.00004,
               alpha=1.0, num_rgb_layer=0, norm_type='bn', **kwargs):
    super().__init__(
      mode, *args,
      weight_decay_scale=weight_decay_scale,
      alpha=alpha,
      num_rgb_layer=num_rgb_layer,
      norm_type=norm_type,
      **kwargs)
    self.mode = mode
    
    if self.mode == 'train':
      _B5 = tf.keras.applications.EfficientNetB5(
        input_shape=[256, 256, 3], include_top=False)
      block3e_add = _B5.layers[188].output  #(32, 32, 64)
      block5g_add = _B5.layers[395].output  #(16, 16, 176)
      block7c_add = _B5.layers[572].output  #(8, 8, 512)
      outputs = [block7c_add, block5g_add, block3e_add]
      self.B5 = tf.keras.Model(_B5.input, outputs, trainable=False)
      self.freeze_model(self.B5)
  
  def freeze_model(self, model: tf.keras.Model):
    for l in model.layers:
      l.trainable = False
      if hasattr(l, 'layers'):
        self.freeze_model(l)
  
  def call(self, inputs, training=None, mask=None):
    raw = inputs[dataset_element.MAI_RAW_PATCH]
    rgb = self._call(raw)
    
    if self.mode == 'train':
      rescale_rgb = tf.clip_by_value(rgb, 0, 1) * 255
      # dslr_rgb = inputs[dataset_element.MAI_DSLR_PATCH]
      block7c_add, block5g_add, block3e_add = self.B5(rescale_rgb)
      
      return {
        model_prediction.ENHANCE_RGB: rgb,
        model_prediction.LARGE_FEAT: block3e_add,
        model_prediction.MID_FEAT: block5g_add,
        model_prediction.SMALL_FEAT: block7c_add,
      }
    else:
      return {
        model_prediction.ENHANCE_RGB: rgb,
      }
