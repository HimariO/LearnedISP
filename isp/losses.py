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