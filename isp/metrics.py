import tensorflow as tf

import abc


class PredictionMetricBase(abc.ABC):

  @abc.abstractmethod
  def __call__(self, name_to_input, name_to_prediction):
    """"""


_TAG_TO_PREDICTION_METRIC = {}


def register_prediction_metric(prediction_metric):
  assert issubclass(prediction_metric, PredictionMetricBase)
  tag = prediction_metric.__name__
  assert tag not in _TAG_TO_PREDICTION_METRIC
  _TAG_TO_PREDICTION_METRIC[tag] = prediction_metric
  return prediction_metric


def get_prediction_metric(tag):
  return _TAG_TO_PREDICTION_METRIC[tag]


def _transform_and_round_to_max_value_255_image(inputs, scale=None, offset=None):
  inputs = tf.cast(inputs, dtype=tf.float32)
  if scale is not None:
    inputs *= scale
  if offset is not None:
    inputs += offset
  return tf.clip_by_value(tf.round(inputs), 0.0, 255.0)


@register_prediction_metric
class PSNR(tf.keras.metrics.Metric, PredictionMetricBase):


  def __init__(self, rgb_label_input_name, rgb_prediction_name):
    super().__init__()
    self._psnr_sum = self.add_weight('psnr_sum', [], initializer='zeros')
    self._psnr_counter = self.add_weight('psnr_counter', [], initializer='zeros')
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    # prediction = tf.clip_by_value(prediction, 0.0, 1.0)
    psnr = tf.image.psnr(y_true, y_pred, 1.0)  # (batch_size, 1)
    
    self._psnr_counter.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    self._psnr_sum.assign_add(tf.reduce_sum(psnr))

  def result(self):
    return self._psnr_sum / self._psnr_counter


@register_prediction_metric
class CacheOutput(tf.keras.metrics.Metric, PredictionMetricBase):

  def __init__(self, **kwargs):
    super().__init__()
    self.y_pred = None
    self.y_ture = None

  def update_state(self, y_true, y_pred, sample_weight=None):
    self.y_pred = y_pred
    self.y_true = y_true

  def result(self):
    if self.y_pred is not None and self.y_true is not None:
      return [
        self.y_pred, 
        self.y_true,
      ]
    else:
      return 0.0


class Count(tf.keras.metrics.Metric, PredictionMetricBase):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._sum = self.add_weight('sum', [], initializer='zeros')
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    self._sum.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
  
  def result(self):
    return self._sum