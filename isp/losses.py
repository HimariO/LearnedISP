import abc
import tensorflow as tf

class PredictionLossBase(abc.ABC, tf.keras.Model):

  def __init__(self, *args, **kwargs):
    self.step = tf.Variable(-1, trainable=False, dtype=tf.int64, aggregation=tf.VariableAggregation.MEAN)
    self.last_eval_step = tf.Variable(-1, trainable=False, dtype=tf.int64, aggregation=tf.VariableAggregation.MEAN)
    
    if 'custom_call' in kwargs:
      self._custom_call = kwargs['custom_call']
      del kwargs['custom_call']
    else:
      self._custom_call = True
    super(PredictionLossBase, self).__init__(*args, **kwargs)

  @abc.abstractmethod
  def __call__(self, name_to_input, name_to_prediction):
    insure_updated = tf.assert_greater(
        self.step, self.last_eval_step, message='Stepping to update steps before clac loss value!')
    with tf.control_dependencies([insure_updated]):
      self.last_eval_step.assign(self.step)
  
  def stepping(self, step):
    assert step >= 0
    self.step.assign(step)


_TAG_TO_PREDICTION_LOSS = {}


def register_prediction_loss(prediction_loss):
  # assert issubclass(prediction_loss, PredictionLossBase)
  tag = prediction_loss.__name__
  assert tag not in _TAG_TO_PREDICTION_LOSS
  _TAG_TO_PREDICTION_LOSS[tag] = prediction_loss
  return prediction_loss


def get_prediction_loss(tag):
  return _TAG_TO_PREDICTION_LOSS[tag]


"""
Keras Loss Classes
"""


@register_prediction_loss
class L2Loss(PredictionLossBase):

  def __init__(self, label_input_name, prediction_name):
    super().__init__()
    self._label_input_name = label_input_name
    self._prediction_name = prediction_name

  def __call__(self, name_to_input, name_to_prediction):
    return (tf.nn.l2_loss(name_to_input[self._label_input_name] -
                          name_to_prediction[self._prediction_name]) /
            tf.cast(tf.size(input=name_to_prediction[self._prediction_name]), dtype=tf.float32))


@register_prediction_loss
class L1Loss(PredictionLossBase):

  def __init__(self, label_input_name, prediction_name):
    super().__init__()
    self._label_input_name = label_input_name
    self._prediction_name = prediction_name

  def __call__(self, name_to_input, name_to_prediction):
    return tf.reduce_mean(
      tf.abs(
        name_to_input[self._label_input_name] -
        name_to_prediction[self._prediction_name]))


@register_prediction_loss
class MSSSIM(tf.keras.losses.Loss):

  # def __init__(self, label_input_name, prediction_name):
  #   super().__init__()
  #   self._label_input_name = label_input_name
  #   self._prediction_name = prediction_name

  def call(self, y_true, y_pred):
    # import pdb; pdb.set_trace()
    return 1 - tf.image.ssim_multiscale(
        y_true,
        y_pred,
        1.0)


@register_prediction_loss
class HypbirdSSIM(tf.keras.losses.Loss):

  def call(self, y_true, y_pred):
    # import pdb; pdb.set_trace()
    ms_ssim = 1 - tf.image.ssim_multiscale(y_true, y_pred, 1.0)
    mse = tf.reduce_mean(tf.keras.losses.mse(y_true, y_pred))
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    struct_loss = (ms_ssim * 0.8 + l1 * 0.2)
    final_loss = tf.cond(
      mse < 4.0,
      true_fn=lambda: mse + struct_loss,
      false_fn=lambda: mse)
    return final_loss


@register_prediction_loss
class MSE(tf.keras.losses.MeanSquaredError):
  pass


@register_prediction_loss
class ChannelMSE(tf.keras.losses.Loss):

  def __init__(self, target_channel=0):
    super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name='c_mse')
    self.target_channel = target_channel

  def call(self, y_true, y_pred):
    # y_pred = tf.clip_by_value(y_pred, 0, 1)
    # y_true = tf.clip_by_value(y_true, 0, 1)
    c_delta = y_pred[..., self.target_channel] - y_true[..., self.target_channel]
    mse = tf.pow(c_delta, 2)
    return mse


@register_prediction_loss
class ChannelMaxMSE(tf.keras.losses.Loss):

  def call(self, y_true, y_pred):
    # y_pred = tf.clip_by_value(y_pred, 0, 1)
    # y_true = tf.clip_by_value(y_true, 0, 1)
    c_delta = y_pred - y_true
    mse = tf.pow(c_delta, 2)
    max_mse = tf.reduce_max(mse, axis=-1)
    return max_mse


@register_prediction_loss
class SobelMap(tf.keras.losses.Loss):
  
  def call(self, y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 0, 1)
    y_true = tf.clip_by_value(y_true, 0, 1)

    true_edge = tf.image.sobel_edges(y_true)  # (B,H,W,2)
    pred_edge = tf.image.sobel_edges(y_pred)
    
    threshold = tf.reduce_mean(tf.reduce_mean(true_edge, axis=1), axis=1) # (B, 2)
    threshold = (threshold * 0.2)[:, None, None, :]
    threshold_mask = tf.cast(true_edge > threshold, tf.float32)

    mse = tf.pow(true_edge - pred_edge, 2) * threshold_mask
    mse = tf.reduce_mean(mse)
    return mse


@register_prediction_loss
class HypCirdSSIM(tf.keras.losses.Loss):

  def call(self, y_true, y_pred):
    # import pdb; pdb.set_trace()
    c_delta = y_pred - y_true
    mse = tf.pow(c_delta, 2)
    max_mse = tf.reduce_mean(tf.reduce_max(mse, axis=-1))
    
    ms_ssim = 1 - tf.image.ssim_multiscale(y_true, y_pred, 1.0)
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    struct_loss = (ms_ssim * 0.8 + l1 * 0.2)
    
    final_loss = tf.cond(
      max_mse < 4.0,
      true_fn=lambda: max_mse + struct_loss,
      false_fn=lambda: max_mse)
    return final_loss


@register_prediction_loss
class CoBi(tf.keras.losses.Loss):

  def __init__(self, patch_size=[5, 5], sptial_term=0.1, name=None):
    super().__init__(name=name)
    self._patch_size = patch_size
    self._weight_sp = sptial_term
  
  def compute_cx(self, dist_tilde, band_width):
    """
    dist_tilde: (B, H'*W', H'*W')
    """
    w = tf.exp((1 - dist_tilde) / band_width)  # Eq(3)
    cx = w / tf.reduce_sum(w, axis=1, keepdims=True)  # Eq(4)
    return cx
  
  def compute_cos_distance(self, x, y):
    B, H, W, N = tf.shape(x)
    x_vec = tf.reshape(x, [B, -1, N])
    y_vec = tf.reshape(y, [B, -1, N])
    cos_sim = tf.keras.losses.cosine_similarity(x_vec, y_vec)
    raise NotImplementedError('Not done yet!')
  
  def compute_l2_distance(self, x, y):
    """
    x: (B, H', W', Ph*Pw*C)
    """
    B, H, W, N = tf.shape(x)
    x_vec = tf.reshape(x, [B, -1, N])
    y_vec = tf.reshape(y, [B, -1, N])
    x_s = tf.reduce_sum(x_vec ** 2, axis=2, keepdims=True)
    y_s = tf.reduce_sum(y_vec ** 2, axis=2, keepdims=True)

    A = y_vec @ tf.transpose(x_vec, perm=[0, 2, 1])
    dist = y_s - 2 * A + tf.transpose(x_s, perm=[0, 2, 1])
    dist = tf.transpose(dist, perm=[0, 2, 1])
    dist = tf.reshape(dist, [B, H*W, H*W])
    dist = tf.nn.relu(dist)

    return dist
  
  def compute_relative_distance(self, dist_raw):
    """
    dist_raw: [B, H'*W', H'*W']
    """
    dist_min = tf.reduce_min(dist_raw, axis=1, keepdims=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde
  
  def compute_meshgrid(self, inputs):
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

    return tf.concat([grid_x, grid_y], -1)
  
  def call(self, y_true, y_pred):
    """
    y_true: (B, H, W, C) large & pretrained CNN featuremap
    """

    # (B, H, W, C) --> (B, H', W', Ph*Pw*C)  | Patch size: (Ph, Pw)
    tile_true = tf.image.extract_patches(
      y_true,
      sizes=[1, *self._patch_size, 1],
      strides=[1, 1, 1, 1],
      rates=[1, 1, 1, 1],
      padding='VALID'
    )
    tile_pred = tf.image.extract_patches(
      y_pred,
      sizes=[1, *self._patch_size, 1],
      strides=[1, 1, 1, 1],
      rates=[1, 1, 1, 1],
      padding='VALID'
    )

    grid = self.compute_meshgrid(tile_pred)
    dist_raw = self.compute_l2_distance(grid, grid)
    dist_tilde = self.compute_relative_distance(dist_raw)
    cx_sp = self.compute_cx(dist_tilde, 1.0)

    # feature loss
    dist_raw = self.compute_l2_distance(tile_true, tile_pred)
    dist_tilde = self.compute_relative_distance(dist_raw)
    cx_feat = self.compute_cx(dist_tilde, 1.0)

    # combined loss
    cx_combine = (1. - self._weight_sp) * cx_feat + self._weight_sp * cx_sp

    k_max_NC = tf.reduce_max(cx_combine, axis=1, keepdims=True)

    cx = tf.reduce_mean(k_max_NC, axis=2)
    cx_loss = tf.reduce_mean(-tf.math.log(cx + 1e-5))

    return cx_loss

    