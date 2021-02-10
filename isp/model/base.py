import tensorflow as tf

from .io import model_prediction, dataset_element


class RawBase(tf.keras.models.Model):

  def output_to_uint8_rgb(self, output_tensor):
    scaled_label = (output_tensor + 1) / 2 * 255
    scaled_label = tf.clip_by_value(scaled_label, 0, 255)
    scaled_label = tf.cast(scaled_label, tf.uint8)
    return scaled_label
  
  def summary_output(self, step, name_to_pred, summary_writer, name_to_label=None):
    if step % 200 != 0:
      return
    pred_summary_name = 'night_sight_prediction' if self._mode != 'eval' else 'eval/night_sight_prediction'
    label_summary_name = 'night_sight_ground_truth' if self._mode != 'eval' else 'eval/night_sight_ground_truth'
    with summary_writer.as_default():
      norm_rgb_tensor = name_to_pred[model_prediction.ENHANCE_RGB]
      scaled_rgb = self.output_to_uint8_rgb(norm_rgb_tensor)
      tf.summary.image(pred_summary_name, scaled_rgb, step=step)
    
      if name_to_label:
        norm_label = name_to_label[dataset_element.MAI_DSLR_PATCH]
        scaled_label = self.output_to_uint8_rgb(norm_label)
        tf.summary.image(label_summary_name, scaled_label, step=step)
  

_TAG_TO_PREDICTION_MODEL = {}


def register_model(prediction_model):
  # assert issubclass(prediction_model, RawBase)
  tag = prediction_model.__name__
  assert tag not in _TAG_TO_PREDICTION_MODEL
  _TAG_TO_PREDICTION_MODEL[tag] = prediction_model
  return prediction_model


def get_model(tag):
  return _TAG_TO_PREDICTION_MODEL[tag]