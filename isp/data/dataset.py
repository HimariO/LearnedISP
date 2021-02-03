import abc
import collections

import tensorflow as tf

from . import aug
from typing import Sized


TF_RECORD_SUFFIX = '.tfrecord'

TFExampleFeature = collections.namedtuple(
    'TFExampleFeature',
    ['key', 'to_example_feature_fn', 'feature_configuration'])

# NOTE: SID_RAW_INPUT contain PNG bytes string, not encoded JPEG byte string!
SID_RAW_INPUT = TFExampleFeature(
    'sid/raw_input',
    lambda value: tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])),
    tf.io.FixedLenFeature([], tf.string)
)
# NOTE: height & width saved over here is size of raw image before rehsape 
#       to [h/2, w/2, 4] using space_to_depth function
SID_RAW_INPUT_HEIGHT = TFExampleFeature(
    'sid/raw_input/height',
    lambda value: tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value])),
    tf.io.FixedLenFeature([], tf.int64)
)
SID_RAW_INPUT_WIDTH = TFExampleFeature(
    'sid/raw_input/width',
    lambda value: tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value])),
    tf.io.FixedLenFeature([], tf.int64)
)
# NOTE: camera exposure time mesuare in second
SID_RAW_INPUT_EXPOSURE=TFExampleFeature(
    'sid/raw_input/exposure_time',
    lambda value: tf.train.Feature(
        float_list=tf.train.FloatList(value=[value])),
    tf.io.FixedLenFeature([], tf.float32)
)
SID_RAW_INPUT_BLACK_LEVEL = TFExampleFeature(
    'sid/raw_input/black_level',
    lambda value: tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value])),
    tf.io.FixedLenFeature([], tf.int64)
)
SID_RGB_GROUND_TRUTH=TFExampleFeature(
    'sid/ground_truth/encoded',
    lambda value: tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])),
    tf.io.FixedLenFeature([], tf.string))
SID_RGB_GROUND_TRUTH_EXPOSURE = TFExampleFeature(
    'sid/ground_truth/exposure_time',
    lambda value: tf.train.Feature(
        float_list=tf.train.FloatList(value=[value])),
    tf.io.FixedLenFeature([], tf.float32)
)


class Base(abc.ABC):

  @abc.abstractmethod
  def create_dataset(self, batch_size=None, num_epochs=None, shuffle=True):
    """"""

_TAG_TO_DATASET = {}


def register_dataset(dataset):
  assert issubclass(dataset, Base)
  tag = dataset.__name__
  assert tag not in _TAG_TO_DATASET
  _TAG_TO_DATASET[tag] = dataset
  return dataset


def get_dataset(tag):
  return _TAG_TO_DATASET[tag]


class TFRecordDataset(Base):

  def __init__(self, data_preprocessing_tag_and_init_kwargs_pairs=None):
    self._data_preprocessing_callables = [
        aug.get_data_preprocessing_callable(name)(**init_kwargs)
        for name, init_kwargs in
        (data_preprocessing_tag_and_init_kwargs_pairs or [])
    ]

  @abc.abstractmethod
  def _parse_tf_record(self, record):
    """"""

  @property
  @abc.abstractmethod
  def tf_record_paths(self) -> Sized:
    """"""

  def _preprocess(self, input_name_to_tensor):
    for data_preprocessing_callable in self._data_preprocessing_callables:
      input_name_to_tensor = data_preprocessing_callable(input_name_to_tensor)
    return input_name_to_tensor

  def _create_example_dataset(self, num_readers, shuffle, cache_examples):
    # NOTE: num_reader > 1 will make output examples become somewhat random due to async reading
    # assert shuffle or num_readers == 1
    tf_record_paths = self.tf_record_paths
    assert tf_record_paths
    example_dataset = tf.data.Dataset.from_tensor_slices(tf_record_paths)
    if shuffle:
      example_dataset = example_dataset.shuffle(len(tf_record_paths))
    example_dataset = example_dataset.apply(
        tf.data.experimental.parallel_interleave(
            lambda filename: tf.data.TFRecordDataset(
                filename, buffer_size=8 * 1024 * 1024),
            cycle_length=num_readers, sloppy=True))
    if cache_examples:
      example_dataset = example_dataset.cache()
    return example_dataset

  def create_dataset(self, batch_size=None, num_epochs=None, shuffle=True,  # pylint: disable=arguments-differ
                     num_readers=1, shuffle_buffer_size=10000,
                     num_parallel_calls=1, drop_remainder=False,
                     cache_examples=False):
    dataset = self._create_example_dataset(num_readers, shuffle, cache_examples)
    if shuffle:
      dataset = dataset.apply(
          tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size,
                                             count=num_epochs))
    else:
      dataset = dataset.repeat(count=num_epochs)

    def parse_and_preprocess_record(record):
      input_name_to_tensor = self._parse_tf_record(record)
      return self._preprocess(input_name_to_tensor)

    if batch_size is None:
      dataset = dataset.map(parse_and_preprocess_record,
                            num_parallel_calls=num_parallel_calls)
    else:
      drop_remainder = ((batch_size == 1) or (num_epochs is None) or
                        drop_remainder)
      dataset = dataset.apply(
          tf.data.experimental.map_and_batch(
              parse_and_preprocess_record, batch_size,
              drop_remainder=drop_remainder,
              num_parallel_calls=num_parallel_calls))
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)
