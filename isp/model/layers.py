import functools

import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow import keras
# from tensorflow.keras.layers import BatchNormalization

# BatchNormalization = functools.partial(
#   tf.keras.layers.Lambda,
#   lambda x: tf.contrib.slim.batch_norm(x, is_training=True))

# def BatchNormalization(is_training=True, **kwargs):
#   # return tf.keras.layers.Lambda(lambda x: tf.contrib.slim.batch_norm(x, is_training=is_training), **kwargs)
#   func = lambda x: tf.compat.v1.layers.batch_normalization(x, training=is_training, fused=False)
#   return tf.keras.layers.Lambda(func, **kwargs)


class BatchNormalization(tf.keras.layers.BatchNormalization):
  def __init__(self, *args, is_training=True, **kwargs) -> None:
      super().__init__(*args, **kwargs)
      self.is_training = is_training
  
  def call(self, x, training=None):
    return super().call(x, training=self.is_training)
    # return super().call(x, training=None)


try:
  import tensorflow_addons as tfa

  class WeightNormalization(tfa.layers.WeightNormalization):
    """This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction.
    """

    def __init__(self, layer, data_init=True, inference=False, **kwargs):
      super(WeightNormalization, self).__init__(
        layer, data_init=data_init, **kwargs)
      self.inference_mode = inference

    def call(self, inputs):
      """Call `Layer`"""
      if self.inference_mode:
        if not isinstance(self.layer.kernel, tf.Variable):
          self.layer.kernel = tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * self.g
        return self.layer(inputs)
      else:
        def _do_nothing():
          return tf.identity(self.g)

        def _update_weights():
          # Ensure we read `self.g` after _update_weights.
          with tf.control_dependencies(self._initialize_weights(inputs)):
            return tf.identity(self.g)

        g = tf.cond(self._initialized, _do_nothing, _update_weights)

        with tf.name_scope('compute_weights'):
          # Replace kernel by normalized weight variable.
          self.layer.kernel = tf.nn.l2_normalize(
            self.v, axis=self.kernel_norm_axes) * g

          # Ensure we calculate result after updating kernel.
          update_kernel = tf.identity(self.layer.kernel)
          with tf.control_dependencies([update_kernel]):
            outputs = self.layer(inputs)
            return outputs
except ImportError:
  
  class WeightNormalization:
    """
    Dummy for tf 1.5
    """

    def __init__(self, layer, data_init=True, inference=False, **kwargs):
      raise RuntimeError('tf-addons is not supported for version <2.3')


class PySequential:

  def __init__(self, layers=[], name=None):
    self.layers = layers

  def __call__(self, x):
    h = x
    for layer in self.layers:
      if isinstance(layer, tf.keras.Model):
        if hasattr(layer, '_call'):
          logger.warning(f'Flatten keras sub-model: {layer}')
          h = layer._call(h)
        else:
          raise ValueError(f"This model subclass can't be flatten to functional model: {layer}")
      elif isinstance(layer, tf.keras.layers.Layer):
        h = layer(h)
      else:
        raise RuntimeError(f"Object type not supported: {layer}")
    return h


class Polynomial(tf.keras.layers.Layer):

  def __init__(self, *args, **kwargs):
    super().__init__(*args,  **kwargs)
  
  def call(self, x, parameter):
    """
    x: Batched single images [B, H, W, 1] fp32
    parameter: [B, 4] fp32
    """
    y = x * parameter[:, None, None, 0:1]
    y += x**2 * parameter[:, None, None, 1:2]
    y += x**3 * parameter[:, None, None, 2:3]
    y += x**4 * parameter[:, None, None, 3:4]
    return y


class PLCurve(tf.keras.layers.Layer):

  def __init__(self, *args, num_knot=16, **kwargs):
    super().__init__(*args,  **kwargs)
    self.num_knot = num_knot
  
  def call(self, x, parameter):
    """
    x: Batched single images [B, H, W, 1] fp32
    parameter: [B, num_knot] fp32
    """
    param = tf.nn.leaky_relu(parameter)
    slope = param[:, 1:] - param[:, :-1]

    scales = []
    curve_steps = self.num_knot - 1
    for i in range(self.num_knot):
      remain = tf.clip_by_value((x * curve_steps) - i, 0, 1)
      if i > 0:
        scales.append(remain * slope[:, i - 1][:, None, None, None])
      else:
        scales.append(param[:, 0][:, None, None, None] * tf.ones_like(x))
    
    scale_per_pixel = tf.stack(scales, axis=-1)
    scale_per_pixel = tf.reduce_sum(scale_per_pixel, axis=-1)
    y = tf.clip_by_value(x * scale_per_pixel, 0, 1)
    return y


class ConcatCoordinate(tf.keras.layers.Layer):

  def __init__(self, *args, concat_hidden=True, **kwargs):
      super().__init__(*args, **kwargs)
      self.concat_hidden = concat_hidden

  def call(self, inputs, concat_hidden=True):
    add_BC_dims = lambda x: tf.expand_dims(tf.expand_dims(x, axis=0), axis=-1)
    input_shape = tf.shape(input=inputs)

    grid_x, grid_y = tf.meshgrid(tf.range(0, limit=input_shape[2]),
                                tf.range(0, limit=input_shape[1]))

    grid_x = tf.cast(add_BC_dims(grid_x), tf.float32)
    grid_x /= tf.reduce_max(input_tensor=grid_x)
    grid_x = tf.tile(grid_x, [input_shape[0], 1, 1, 1])

    grid_y = tf.cast(add_BC_dims(grid_y), tf.float32)
    grid_y /= tf.reduce_max(input_tensor=grid_y)
    grid_y = tf.tile(grid_y, [input_shape[0], 1, 1, 1])

    if self.concat_hidden:
      output = tf.concat([inputs, grid_x, grid_y], axis=-1)
    else:
      output = tf.concat([grid_x, grid_y], axis=-1)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape[:3] + (input_shape[3] + 2,)


class ResConvBlock(tf.keras.Model):

  def __init__(self, num_layer, channel, weight_norm=True, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._blocks = []
    for i in range(num_layer):
      block = [
        tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=None),
        tf.keras.layers.Activation(tf.nn.relu6),
        BatchNormalization(),
      ]
      # setattr(self, f'_conv_{i}', layer)
      self._blocks.append(block)

  def call(self, inputs):
    x = inputs
    for i, block in enumerate(self._blocks):
      t = x
      for layer in block:
        x = layer(x)
      if t.shape[-1] == x.shape[-1]:
        x += t
    return x


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
      self.bn3 = BatchNormalization(name=f'{self.name}_bn_3', is_training=self.is_training)
      self.bn1 = BatchNormalization(name=f'{self.name}_bn_1', is_training=self.is_training)
      # self.bn_id = BatchNormalization(name=f'{self.name}_bn_id')
      self.bn_id = lambda x: x
    elif self.norm_type == 'wn':
      self.conv3 = WeightNormalization(self.conv3)
      self.conv1 = WeightNormalization(self.conv1)
      
      self.bn3 = lambda x: x
      self.bn1 = lambda x: x
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
      
      kid_init = np.zeros([3, 3, self.channel, self.channel], dtype=np.float32)
      if int(k1.shape[2]) == int(k1.shape[3]):
        for c in range(self.channel):
          kid_init[1, 1, c, c] = 1.0
      kid = tf.constant(kid_init)
      k_rep = k3 + k1_3 + kid
      b_rep = b3 + b1

      return k_rep, b_rep
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


class _RepConv(tf.keras.layers.Layer):

  def __init__(self,
              output_channel,
              filter_size,
              padding='same',
              strides=(1, 1),
              activation=tf.nn.relu, 
              inference=False,
              norm_type='bn',
              trainable=True,
              **kwargs):
    super().__init__()
    self.channel = output_channel
    self.filter_size = filter_size
    self.padding = padding
    self.strides = strides
    self.activation = activation
    self.inference = inference
    self.norm_type = norm_type
    self.trainable = trainable
    self._build()
  
  def build(self, input_shape):
    
    Ci = input_shape[-1]
    Co = self.channel
    self.w3 = self.add_weight(
      name='conv3_w',
      shape=[self.filter_size, self.filter_size, Ci, Co],
      trainable=self.trainable)
    self.w1 = self.add_weight(
      name='conv1_w',
      shape=[1, 1, Ci, Co],
      trainable=self.trainable)
    
    if self.norm_type == 'bn':
      self.bn3 = BatchNormalization()
      self.bn1 = BatchNormalization()
      # self.bn_id = BatchNormalization()
      self.bn_id = lambda x: x
    else:
      raise ValueError(f"Unknow normalization type: {self.norm_type}")

    self.rep_conv3 = tf.keras.layers.Conv2D(
      self.channel, 3, strides=self.strides, padding=self.padding)
  
  def reparameterization(self):
    if self.norm_type == 'bn':
      assert len(self.conv1.get_weights()) == 1
      assert len(self.conv3.get_weights()) == 1
      w1 = self.conv1.get_weights()[0]
      w1_3 = np.pad(w1, [(1, 1), (1, 1), (0, 0), (0, 0)], constant_value=0.0)
      w3 = self.conv3.get_weights()[0]

      raise NotImplementedError()
    else:
      raise ValueError(f"Unknow normalization type: {self.norm_type}")

  def call(self, inputs):
    if self.inference:
      return self.rep_conv3(inputs)
    else:
      # x3 = self.bn3(self.conv3_2(self.conv3(self.conv3_1(inputs))))
      x3 = self.bn3(self.conv3(inputs))
      x1 = self.bn1(self.conv1(inputs))
      if int(inputs.shape[-1]) == self.channel:
        identity = self.bn_id(inputs)
        x = self.activation(x1 + x3 + identity)
      else:
        x = self.activation(x1 + x3)
      return x


class UNetBlocks:

  act_func = tf.nn.relu
  def __init__(self, *args, **kwargs) -> None:
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

  def downsample_block(self, channel, name=None, norm_type='bn'):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=None, use_bias=False),
      BatchNormalization(is_training=self.is_training),
      tf.keras.layers.Activation(UNetBlocks.act_func),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=None, use_bias=False),
      BatchNormalization(is_training=self.is_training),
      tf.keras.layers.Activation(UNetBlocks.act_func),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=None, use_bias=False),
      BatchNormalization(is_training=self.is_training),
      tf.keras.layers.Activation(UNetBlocks.act_func),
    ]
    # layers = [WeightNormalization(l, data_init=True, inference=not WN) for l in layers]
    return self.sequential(layers=layers, name=name)
  
  def reverse_downsample_block(self, channel, name=None, norm_type='bn'):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=None, use_bias=False),
      BatchNormalization(is_training=self.is_training),
      tf.keras.layers.Activation(UNetBlocks.act_func),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=None, use_bias=False),
      BatchNormalization(is_training=self.is_training),
      tf.keras.layers.Activation(UNetBlocks.act_func),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=None, use_bias=False),
      BatchNormalization(is_training=self.is_training),
      tf.keras.layers.Activation(UNetBlocks.act_func),
    ]
    return self.sequential(layers=layers, name=name)
  
  def res_downsample_block(self, channel, name=None, norm_type='bn'):
    layers = [
      ResConvBlock(2, channel),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=tf.nn.relu, use_bias=False),
    ]
    # layers = layers[:1] + [WeightNormalization(l, data_init=True, inference=not WN) for l in layers[1:]]
    return self.sequential(layers=layers, name=name)
  
  def reverse_res_downsample_block(self, channel, name=None, norm_type='bn'):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=tf.nn.relu, use_bias=False),
      ResConvBlock(2, channel),
    ]
    # layers = [WeightNormalization(l, data_init=True, inference=not WN) for l in layers[:-1]] + layers[-1:]
    return self.sequential(layers=layers, name=name)
  
  def conv_block(self, channel, name=None, norm_type='bn'):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), use_bias=False),
      BatchNormalization(is_training=self.is_training),
      tf.keras.layers.Activation(UNetBlocks.act_func),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), use_bias=False),
      BatchNormalization(is_training=self.is_training),
      tf.keras.layers.Activation(UNetBlocks.act_func),
    ]
    # layers = [WeightNormalization(l, data_init=True, inference=not WN) for l in layers]
    return self.sequential(layers=layers, name=name)
  
  def res_conv_block(self, channel, name=None, norm_type='bn'):
    return ResConvBlock(2, channel)
  
  def upsample_block(self, channel, name=None, norm_type='bn'):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), use_bias=False),
      BatchNormalization(is_training=self.is_training),
      tf.keras.layers.Activation(UNetBlocks.act_func),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), use_bias=False),
      BatchNormalization(is_training=self.is_training),
      tf.keras.layers.Activation(UNetBlocks.act_func),
      tf.keras.layers.Conv2DTranspose(channel // 2, 3, padding='same', strides=(2, 2), activation=None, use_bias=False),
    ]
    # layers = [WeightNormalization(l, data_init=True, inference=not WN) for l in layers[:-1]] + [layers[-1]]
    return self.sequential(layers=layers, name=name)

  def upsample_layer(self, channel, name=None, norm_type='bn'):
    layer = tf.keras.layers.Conv2DTranspose(channel, 3, padding='same', strides=(2, 2), use_bias=False)
    return layer
  
  def res_upsample_block(self, channel, name=None, norm_type='bn'):
    layers = [
      ResConvBlock(2, channel, weight_norm=True),
      tf.keras.layers.Conv2DTranspose(channel // 2, 3, padding='same', strides=(2, 2)),
    ]
    return self.sequential(layers=layers, name=name)
  
  def rgb_upsample_block(self, num_rgb_layer=0, name=None):
    layers = [
        tf.keras.layers.Conv2DTranspose(
            12, 3, padding='same', strides=(2, 2), activation=tf.nn.relu),
    ]
    layers += [tf.keras.layers.Conv2D(12, 3, padding='same', activation=UNetBlocks.act_func) for _ in range(num_rgb_layer)]
    layers += [tf.keras.layers.Conv2D(3, 1)]
    return self.sequential(layers=layers, name=name)


class UNetBilinearBlocks(UNetBlocks):

  def upsample_block(self, channel, name=None, norm_type='bn'):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), use_bias=False),
      BatchNormalization(is_training=self.is_training),
      tf.keras.layers.Activation(UNetBlocks.act_func),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), use_bias=False),
      BatchNormalization(is_training=self.is_training),
      tf.keras.layers.Activation(UNetBlocks.act_func),
      tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
      tf.keras.layers.Conv2D(channel // 2, 3, padding='same', strides=(1, 1), use_bias=False),
      BatchNormalization(is_training=self.is_training),
    ]
    
    # layers = [
    #   WeightNormalization(l, data_init=True, inference=not WN)
    #   for l in layers[:-1]
    # ] + [layers[-1]]
    return self.sequential(layers=layers, name=name)

  def upsample_layer(self, channel, name=None, norm_type='bn'):
    layers = [
      tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), use_bias=False),
    ]
    return self.sequential(layers=layers, name=name)
  
  def res_upsample_block(self, channel, name=None, norm_type='bn'):
    layers = [
      ResConvBlock(2, channel),
      tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
      tf.keras.layers.Conv2D(channel // 2, 3, padding='same', strides=(1, 1), activation=None, use_bias=False),
    ]
    return self.sequential(layers=layers, name=name)
  
  def rgb_upsample_block(self, num_rgb_layer=1, name=None, norm_type='bn'):
    layers = [
        tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
    ]
    for _ in range(num_rgb_layer):
      layers += [
        tf.keras.layers.Conv2D(8, 3, padding='same', use_bias=False),
        BatchNormalization(is_training=self.is_training),
        tf.keras.layers.Activation(UNetBlocks.act_func),
      ]
    layers += [tf.keras.layers.Conv2D(3, 1)]
    return self.sequential(layers=layers, name=name)


class RepVGGBlocks:

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
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training),
      RepConv(channel, 3, padding='same', strides=(2, 2), activation=self.block_act, norm_type=norm_type, is_training=self.is_training),
    ]
    return self.sequential(layers=layers, name=name)
  
  def reverse_downsample_block(self, channel, name=None, norm_type='wn'):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=None),
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training),
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training),
    ]
    return self.sequential(layers=layers, name=name)
  
  def res_downsample_block(self, channel, name=None, norm_type='wn'):
    layers = [
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training),
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=None),
    ]
    return self.sequential(layers=layers, name=name)
  
  def reverse_res_downsample_block(self, channel, name=None, norm_type='wn', num_block=2):
    layers = [
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(2, 2), activation=None)] + [
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training)
      for _ in range(num_block)
    ]
    return self.sequential(layers=layers, name=name)
  
  def conv_block(self, channel, name=None, norm_type='wn', num_block=2):
    layers = [
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training)
      for _ in range(num_block)
    ]
    return self.sequential(layers=layers, name=name)
  
  def res_conv_block(self, channel, name=None, norm_type='wn', num_block=2):
    return self.conv_block(channel, name=name, norm_type=norm_type, num_block=num_block)
  
  def upsample_block(self, channel, name=None, norm_type='wn'):
    layers = [
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training),
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training),
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
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training),
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
      RepConv(12, 3, padding='same', strides=(1, 1), activation=None, norm_type=norm_type, is_training=self.is_training)
      for _ in range(num_rgb_layer)
    ]
    layers += [tf.keras.layers.Conv2D(3, 1)]
    return self.sequential(layers=layers, name=name)


class RepBilinearVGGBlocks(RepVGGBlocks):

  def upsample_block(self, channel, name=None, norm_type='wn'):
    layers = [
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training),
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training),
      tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
      RepConv(channel // 2, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training),
    ]
    return self.sequential(layers=layers, name=name)

  def upsample_layer(self, channel, name=None, norm_type='wn'):
    layers = [
      tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
      RepConv(channel, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training),
    ]
    return self.sequential(layers=layers, name=name)
  
  def res_upsample_block(self, channel, name=None, norm_type='wn'):
    layers = [
      RepConv(channel, 3, padding='same', strides=(1, 1), norm_type=norm_type, is_training=self.is_training, activation=self.block_act),
      RepConv(channel, 3, padding='same', strides=(1, 1), norm_type=norm_type, is_training=self.is_training, activation=self.block_act),
      tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
      RepConv(channel // 2, 3, padding='same', strides=(1, 1), activation=self.block_act, norm_type=norm_type, is_training=self.is_training),
    ]
    return self.sequential(layers=layers, name=name)
  
  def rgb_upsample_block(self, num_rgb_layer=1, name=None, norm_type='wn'):
    layers = [
        tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
    ]
    layers += [
      RepConv(8, 3, padding='same', activation=None, norm_type=norm_type, is_training=self.is_training)
      for _ in range(num_rgb_layer)]
    layers += [tf.keras.layers.Conv2D(3, 1)]
    return self.sequential(layers=layers, name=name)