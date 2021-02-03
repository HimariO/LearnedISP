import abc
import collections
import tensorflow as tf

from ..model.io import dataset_element

class DataPreprocessingBase(abc.ABC):

  @abc.abstractmethod
  def __call__(self, input_name_to_tensor):
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



"""
All avaliable preprocessing OP/augmentations
"""

@register_preprocessing_callable
class NormalizeUint8ImageIntensity(DataPreprocessingBase):

  def __init__(self, scale=2.0 / 255.0, offset=-128 * 2.0 / 255.0,
               images_are_0_to_255_float=False):
    super().__init__()
    self._scale = scale
    self._offset = offset
    self._images_are_0_to_255_float = images_are_0_to_255_float

  def __call__(self, input_name_to_tensor):
    for image_name in [dataset_element.RGB_IMAGE]:
      if image_name not in input_name_to_tensor:
        continue

      image = input_name_to_tensor[image_name]
      if not self._images_are_0_to_255_float:
        image = tf.image.convert_image_dtype(image, tf.uint8)
        image = tf.cast(image, dtype=tf.float32)
      image = self._scale * image + self._offset
      input_name_to_tensor[image_name] = image
    return input_name_to_tensor


@register_preprocessing_callable
class RandomRot90(DataPreprocessingBase):

  def _rot90(predicate, image):
    image_shape = image.shape
    image = tf.image.rot90(image, tf.cast(predicate, dtype=tf.int32))
    if (image_shape[-3:-1].is_fully_defined() and
        (image_shape[-3].value == image_shape[-2].value)):
      image.set_shape(image_shape)
    return image

  def __call__(self, input_name_to_tensor):
    k = tf.random.uniform([], minval=0.0, maxval=1.0) * 4
    for image_name in [
      dataset_element.RGB_IMAGE,
      dataset_element.MAI_RAW_PATCH,
      dataset_element.MAI_DSLR_PATCH,
    ]:
      image = input_name_to_tensor.get(image_name)
      if image is None:
        continue

      image = self._rot90(k, image)
      input_name_to_tensor[image_name] = image
    return input_name_to_tensor