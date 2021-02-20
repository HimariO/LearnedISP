import tensorflow as tf
import tensorflow_addons as tfa


class PySequential:

  def __init__(self, layers=[], name=None):
    self.layers = layers

  def __call__(self, x):
    h = x
    for layer in self.layers:
      h = layer(h)
    return h


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

  def call(self, inputs, **kwargs):
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

    output = tf.concat([inputs, grid_x, grid_y], axis=-1)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape[:3] + (input_shape[3] + 2,)


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
    return tf.keras.Sequential(layers=layers, name=name)

  def upsample_layer(self, channel, name=None, WN=True):
    layers = [
      tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
      tf.keras.layers.Conv2D(channel, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
    ]
    return tf.keras.Sequential(layers)
  
  def res_upsample_block(self, channel, name=None, WN=True):
    layers = [
      ResConvBlock(2, channel, weight_norm=WN),
      tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
      tf.keras.layers.Conv2D(channel // 2, 3, padding='same', strides=(1, 1), activation=tf.nn.relu),
    ]
    return tf.keras.Sequential(layers=layers, name=name)
  
  def rgb_upsample_block(self, num_rgb_layer=1, name=None):
    layers = [
        tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
    ]
    layers += [tf.keras.layers.Conv2D(8, 3, padding='same', activation=tf.nn.relu) for _ in range(num_rgb_layer)]
    layers += [tf.keras.layers.Conv2D(3, 1)]
    return tf.keras.Sequential(layers=layers, name=name)