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

  COLOR_SPACE_Y = 'Y'
  COLOR_SPACE_RGB = 'RGB'

  def __init__(self, rgb_label_input_name, rgb_prediction_name, scale=None,
         offset=None, border=0, color_space=None, uint8_image=False):
    super().__init__()
    self._rgb_label_input_name = rgb_label_input_name
    self._rgb_prediction_name = rgb_prediction_name
    self._scale = scale
    self._offset = offset
    self._border = border
    self._uint8_image = uint8_image
    
    color_space = color_space or type(self).COLOR_SPACE_Y
    assert color_space in {type(self).COLOR_SPACE_Y, type(self).COLOR_SPACE_RGB}
    self._color_space = color_space

    self._psnr_sum = self.add_weight('psnr_sum', [], initializer='zeros')
    self._psnr_counter = self.add_weight('psnr_counter', [], initializer='zeros')

  def __call__(self, name_to_input, name_to_prediction):
    self.update_state(name_to_input, name_to_prediction)
    return self.result
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    # We align the order of rounding and type casting with EDSR-PyTorch.
    if self._uint8_image:
      label = _transform_and_round_to_max_value_255_image(
        y_true, scale=self._scale,
        offset=self._offset) / 255.0
      prediction = _transform_and_round_to_max_value_255_image(
        y_pred, scale=self._scale,
        offset=self._offset) / 255.0
    else:
      label = y_true
      prediction = y_pred

    if self._color_space == type(self).COLOR_SPACE_Y:
      label = tf.tensordot(
        label, [65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0],
        axes=[-1, -1])
      prediction = tf.tensordot(
        prediction,
        [65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0], axes=[-1, -1])
      # The last three dimensions of tf.image.psnr inputs are expected to be
      # [height, width, depth].
      label = label[..., None]
      prediction = prediction[..., None]

    if self._border:
      roi_slice = slice(self._border, -self._border)
      label = label[..., roi_slice, roi_slice, :]
      prediction = prediction[..., roi_slice, roi_slice, :]
    
    prediction = tf.clip_by_value(prediction, 0.0, 1.0)
    psnr = tf.image.psnr(label, prediction, 1.0)  # (batch_size, 1)
    
    self._psnr_counter.assign_add(tf.cast(tf.shape(psnr)[0], tf.float32))
    self._psnr_sum.assign_add(tf.reduce_sum(psnr))

  def result(self):
    return self._psnr_sum / self._psnr_counter
