import abc

import tensorflow as tf
from loguru import logger
from ..model import io

class DataPreprocessingBase(abc.ABC):

  @abc.abstractmethod
  def __call__(self, input_and_label):
    """"""


_TAG_TO_PREPROCESSING_CALLABLE = {}


def register_preprocessing_callable(preprocessing_callable):
  assert issubclass(preprocessing_callable, DataPreprocessingBase)
  tag = preprocessing_callable.__name__
  assert tag not in _TAG_TO_PREPROCESSING_CALLABLE
  _TAG_TO_PREPROCESSING_CALLABLE[tag] = preprocessing_callable
  return preprocessing_callable


def get_data_preprocessing_callable(tag):
  return _TAG_TO_PREPROCESSING_CALLABLE[tag]


@register_preprocessing_callable
class RandomFlip(DataPreprocessingBase):

  def __init__(self, top_down=False):
    super().__init__()
    self._top_down = top_down

  def __call__(self, input_and_label):
    input_dict, label_dict = input_and_label
    to_flip = tf.random.uniform([], minval=0.0, maxval=1.0) <= 0.5
    
    for image_name, label_name, width_axis in [
        (io.dataset_element.MAI_RAW_PATCH, io.model_prediction.ENHANCE_RGB, 1)]:
      image = input_dict.get(image_name)
      if image is None:
        continue

      image = tf.cond(
        to_flip,
        true_fn=lambda: (
          tf.image.flip_up_down(image) 
          if self._top_down 
          else tf.image.flip_left_right(image)
        ),
        false_fn=lambda: image,
      )
      input_dict[image_name] = image
      
      label = label_dict[label_name]
      # NOTE: apply only if label if a image or mask
      if len(label.shape) >= 3:
        logger.debug("RandomFlip")
        label = tf.cond(
          to_flip,
          true_fn=lambda: (
            tf.image.flip_up_down(label) 
            if self._top_down 
            else tf.image.flip_left_right(label)
          ),
          false_fn=lambda: label,
        )
        label_dict[label_name] = label
    
    return input_dict, label_dict


@register_preprocessing_callable
class InsertGrayscale(DataPreprocessingBase):
  """
  Create grayscale version of groundtruth image for two-tage ISP model
  NOTE: this preprocessor must be put in the end of the pipeline
  """

  def __init__(self):
    super().__init__()

  def __call__(self, input_and_label):
    logger.debug("InsertGrayscale")
    input_dict, label_dict = input_and_label
    
    if io.model_prediction.ENHANCE_RGB in label_dict:
      img = label_dict[io.model_prediction.ENHANCE_RGB]
      label_dict[io.model_prediction.INTER_MID_GRAY] = tf.reduce_mean(img, axis=-1, keepdims=True)
    
    if io.dataset_element.MAI_DSLR_PATCH in label_dict:
      img = label_dict[io.dataset_element.MAI_DSLR_PATCH]
      label_dict[io.dataset_element.MAI_DSLR_GRAY_PATCH] = tf.reduce_mean(img, axis=-1, keepdims=True)
    
    return input_dict, label_dict