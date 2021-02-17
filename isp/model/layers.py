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