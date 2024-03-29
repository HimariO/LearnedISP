import abc
import collections
from typing import Sized

import tensorflow as tf
from loguru import logger

from . import aug
from isp.model.io import dataset_element, model_prediction


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


MAI_SAMPLE_ID = TFExampleFeature(
    'mai/id',
    lambda value: tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value])),
    tf.io.FixedLenFeature([], tf.int64)
)
MAI_RAW_INPUT = TFExampleFeature(
    'mai/raw_input',
    lambda value: tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])),
    tf.io.FixedLenFeature([], tf.string)
)
# NOTE: height & width saved over here is size of raw image before rehsape 
#       to [h/2, w/2, 4] using space_to_depth function
MAI_RAW_INPUT_HEIGHT = TFExampleFeature(
    'mai/raw_input/height',
    lambda value: tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value])),
    tf.io.FixedLenFeature([], tf.int64)
)
MAI_RAW_INPUT_WIDTH = TFExampleFeature(
    'mai/raw_input/width',
    lambda value: tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value])),
    tf.io.FixedLenFeature([], tf.int64)
)
MAI_RGB_GROUND_TRUTH=TFExampleFeature(
    'mai/ground_truth/encoded',
    lambda value: tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])),
    tf.io.FixedLenFeature([], tf.string))

"""
CNN feature maps for CoBi Contextual Loss
"""
MAI_RGB_B5_3E = TFExampleFeature(
    'mai/image_feature/block3e_add/float_list',
    lambda value: tf.train.Feature(
        float_list=tf.train.FloatList(value=value)),
    tf.io.VarLenFeature(tf.float32))
MAI_RGB_B5_3E_SHAPE = TFExampleFeature(
    'mai/image_feature/block3e_add/shape',
    lambda value: tf.train.Feature(
        int64_list=tf.train.Int64List(value=value)),
    tf.io.FixedLenFeature([4], tf.int64))
MAI_RGB_B5_5G = TFExampleFeature(
    'mai/image_feature/block5g_add/float_list',
    lambda value: tf.train.Feature(
        float_list=tf.train.FloatList(value=value)),
    tf.io.VarLenFeature(tf.float32))
MAI_RGB_B5_5G_SHAPE = TFExampleFeature(
    'mai/image_feature/block5g_add/shape',
    lambda value: tf.train.Feature(
        int64_list=tf.train.Int64List(value=value)),
    tf.io.FixedLenFeature([4], tf.int64))
MAI_RGB_B5_7C = TFExampleFeature(
    'mai/image_feature/block7c_add/float_list',
    lambda value: tf.train.Feature(
        float_list=tf.train.FloatList(value=value)),
    tf.io.VarLenFeature(tf.float32))
MAI_RGB_B5_7C_SHAPE = TFExampleFeature(
    'mai/image_feature/block7c_add/shape',
    lambda value: tf.train.Feature(
        int64_list=tf.train.Int64List(value=value)),
    tf.io.FixedLenFeature([4], tf.int64))


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


def split_train_valid(dataset: tf.data.Dataset, val_ratio_or_size=0.1):
  if val_ratio_or_size >= 1:
    val_set = dataset.take(val_ratio_or_size)
    train_set = dataset.skip(val_ratio_or_size)
  else:
    N = tf.data.cardinality(dataset)
    V = int(val_ratio_or_size * N)
    val_set = dataset.take(V)
    train_set = dataset.skip(V)
  
  return (
    train_set.repeat().prefetch(tf.data.AUTOTUNE),
    val_set.prefetch(tf.data.AUTOTUNE))


class TFRecordDataset(Base):

  def __init__(self, data_preprocessing_callables=[]):
    self._data_preprocessing_callables = data_preprocessing_callables

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

  # def _create_example_dataset(self, num_readers, shuffle, cache_examples):
  #   # NOTE: num_reader > 1 will make output examples become somewhat random due to async reading
  #   # assert shuffle or num_readers == 1
  #   tf_record_paths = self.tf_record_paths
  #   assert tf_record_paths
  #   example_dataset = tf.data.Dataset.from_tensor_slices(tf_record_paths)
  #   if shuffle:
  #     example_dataset = example_dataset.shuffle(len(tf_record_paths))
  #   example_dataset = example_dataset.apply(
  #       tf.data.experimental.parallel_interleave(
  #           lambda filename: tf.data.TFRecordDataset(
  #               filename, buffer_size=8 * 1024 * 1024),
  #           cycle_length=num_readers, sloppy=True))
  #   if cache_examples:
  #     example_dataset = example_dataset.cache()
  #   return example_dataset
  
  def _create_example_dataset(self, num_readers, shuffle, cache_examples):
    # NOTE: num_reader > 1 will make output examples become somewhat random due to async reading
    # assert shuffle or num_readers == 1
    tf_record_paths = self.tf_record_paths
    assert tf_record_paths
    example_dataset = tf.data.TFRecordDataset(tf_record_paths, num_parallel_reads=num_readers)
    
    if shuffle:
      example_dataset = example_dataset.shuffle(4096)
    if cache_examples:
      example_dataset = example_dataset.cache()
    
    return example_dataset

  def create_dataset(self, batch_size=None, num_epochs=None, shuffle=True,  # pylint: disable=arguments-differ
                     num_readers=1, shuffle_buffer_size=4096,
                     num_parallel_calls=1, drop_remainder=False,
                     cache_examples=False):
    dataset = self._create_example_dataset(num_readers, shuffle, cache_examples)
    # dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # if shuffle:
    #   dataset = dataset.apply(
    #       tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size,
    #                                          count=num_epochs))
    # else:
    #   dataset = dataset.repeat(count=num_epochs)

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
    return dataset


@register_dataset
class SIDTFRecordDataset(TFRecordDataset):

  BIT_DEPTH = 16383.0 

  def __init__(self, data_preprocessing_callables=[],
               tf_record_path_pattern=None, is_verification_dataset=False):
    super().__init__(
        data_preprocessing_callables=(
            data_preprocessing_callables))
    assert tf_record_path_pattern is not None
    self._tf_record_paths = tf.io.gfile.glob(tf_record_path_pattern)
    self._is_verification_dataset = is_verification_dataset

  @property
  def tf_record_paths(self):
    return list(self._tf_record_paths)

  def _normalize_rgb_image(self, img, scale=128, offset=-128):
    return (img + offset) / scale

  def _parse_tf_record(self, record):
    _ = self
    key_to_feature = tf.io.parse_single_example(
        serialized=record,
        features={
            feature.key: feature.feature_configuration
            for feature in [SID_RAW_INPUT,
                            SID_RAW_INPUT_HEIGHT,
                            SID_RAW_INPUT_WIDTH,
                            SID_RAW_INPUT_EXPOSURE,
                            SID_RAW_INPUT_BLACK_LEVEL,
                            SID_RGB_GROUND_TRUTH,
                            SID_RGB_GROUND_TRUTH_EXPOSURE,]
        })
    
    black_level = tf.cast(
        key_to_feature[SID_RAW_INPUT_BLACK_LEVEL.key],
        dtype=tf.float32)
    black_level = tf.reduce_max(black_level)
    
    img_height = tf.cast(key_to_feature[SID_RAW_INPUT_HEIGHT.key], tf.int32)
    img_width = tf.cast(key_to_feature[SID_RAW_INPUT_WIDTH.key], tf.int32)
    
    raw_image = tf.image.decode_png(
        key_to_feature[SID_RAW_INPUT.key],
        channels=1,
        dtype=tf.dtypes.uint16)
    raw_image = tf.nn.space_to_depth(raw_image[tf.newaxis, ...], 2)[0]  # convert flatten png back into 4 channel
    raw_image = tf.cast(raw_image, dtype=tf.float32)
    raw_image = tf.maximum(raw_image - black_level, 0)
    raw_image /= (self.BIT_DEPTH - black_level)
    raw_image.set_shape([None, None, 4])
    # raw_image = tf.image.resize_with_crop_or_pad(raw_image, img_height, img_width)
    # raw_image = (raw_image - 0.5) * 2  # scale from [0, 1] to [-1, 1]

    rgb_ground_truth = tf.image.decode_image(
        key_to_feature[SID_RGB_GROUND_TRUTH.key]
    )
    rgb_ground_truth = tf.cast(rgb_ground_truth, tf.float32)
    rgb_ground_truth = self._normalize_rgb_image(rgb_ground_truth)
    rgb_ground_truth.set_shape([None, None, 3])

    short_exposure = key_to_feature[SID_RAW_INPUT_EXPOSURE.key]
    long_exposure = key_to_feature[SID_RGB_GROUND_TRUTH_EXPOSURE.key]
    ratio = long_exposure / short_exposure

    return {
        dataset_element.SID_RAW_INPUT: raw_image * ratio,
        dataset_element.SID_RGB_GROUND_TRUTH: rgb_ground_truth,
        'black_level': black_level
    }


@register_dataset
class MaiIspTFRecordDataset(TFRecordDataset):

  BIT_DEPTH = 255.0 * 4 *4  # NOTE: raw image actually value from 144 ~ 4092
  BLACK_LEVEL = 144

  def __init__(self, 
      tf_record_path_pattern=None,
      data_preprocessing_callables=[],
      is_verification_dataset=False):
    super().__init__(
        data_preprocessing_callables=(
            data_preprocessing_callables))
    assert tf_record_path_pattern is not None
    self._tf_record_paths = tf.io.gfile.glob(tf_record_path_pattern)
    self._is_verification_dataset = is_verification_dataset

  @property
  def tf_record_paths(self):
    return list(self._tf_record_paths)

  def _normalize_rgb_image(self, img, scale=255, offset=0):
    return (img + offset) / scale

  def _parse_tf_record(self, record):
    _ = self
    key_to_feature = tf.io.parse_single_example(
        serialized=record,
        features={
            feature.key: feature.feature_configuration
            for feature in [MAI_RAW_INPUT,
                            MAI_SAMPLE_ID,
                            MAI_RAW_INPUT_HEIGHT,
                            MAI_RAW_INPUT_WIDTH,
                            MAI_RGB_GROUND_TRUTH,]
        })
    
    sample_id = tf.cast(key_to_feature[MAI_SAMPLE_ID.key], tf.int32)
    img_height = tf.cast(key_to_feature[MAI_RAW_INPUT_HEIGHT.key], tf.int32)
    img_width = tf.cast(key_to_feature[MAI_RAW_INPUT_WIDTH.key], tf.int32)
    
    raw_image = tf.image.decode_png(
        key_to_feature[MAI_RAW_INPUT.key],
        channels=1,
        dtype=tf.dtypes.uint16)
    
    raw_image = tf.nn.space_to_depth(raw_image[tf.newaxis, ...], 2)[0]  # convert flatten png back into 4 channel
    raw_image = tf.cast(raw_image, dtype=tf.float32)
    # tf.debugging.assert_less_equal(raw_image, self.BIT_DEPTH)
    
    raw_image = tf.maximum(raw_image - self.BLACK_LEVEL, 0)
    raw_image /= (self.BIT_DEPTH - self.BLACK_LEVEL)
    raw_image.set_shape([None, None, 4])
    # raw_image = tf.image.resize_with_crop_or_pad(raw_image, img_height, img_width)
    # raw_image = (raw_image - 0.5) * 2  # scale from [0, 1] to [-1, 1]

    rgb_ground_truth = tf.image.decode_image(
        key_to_feature[MAI_RGB_GROUND_TRUTH.key]
    )
    rgb_ground_truth = tf.cast(rgb_ground_truth, tf.float32)
    rgb_ground_truth = self._normalize_rgb_image(rgb_ground_truth)
    rgb_ground_truth.set_shape([None, None, 3])

    return (
      {
        dataset_element.MAI_SMAPLE_ID: sample_id,
        dataset_element.MAI_RAW_PATCH: raw_image,
        dataset_element.MAI_DSLR_PATCH: rgb_ground_truth,
      },
      {
        model_prediction.ENHANCE_RGB: rgb_ground_truth,
      }
    )


@register_dataset
class MaiIspB5TFRecordDataset(MaiIspTFRecordDataset):

  def _parse_tf_record(self, record):
    _ = self
    key_to_feature = tf.io.parse_single_example(
        serialized=record,
        features={
            feature.key: feature.feature_configuration
            for feature in [MAI_RAW_INPUT,
                            MAI_SAMPLE_ID,
                            MAI_RAW_INPUT_HEIGHT,
                            MAI_RAW_INPUT_WIDTH,
                            MAI_RGB_GROUND_TRUTH,
                            MAI_RGB_B5_3E,
                            MAI_RGB_B5_3E_SHAPE,
                            MAI_RGB_B5_5G,
                            MAI_RGB_B5_5G_SHAPE,
                            MAI_RGB_B5_7C,
                            MAI_RGB_B5_7C_SHAPE,]
        })
    
    sample_id = tf.cast(key_to_feature[MAI_SAMPLE_ID.key], tf.int32)
    img_height = tf.cast(key_to_feature[MAI_RAW_INPUT_HEIGHT.key], tf.int32)
    img_width = tf.cast(key_to_feature[MAI_RAW_INPUT_WIDTH.key], tf.int32)
    
    raw_image = tf.image.decode_png(
        key_to_feature[MAI_RAW_INPUT.key],
        channels=1,
        dtype=tf.dtypes.uint16)
    
    raw_image = tf.nn.space_to_depth(raw_image[tf.newaxis, ...], 2)[0]  # convert flatten png back into 4 channel
    raw_image = tf.cast(raw_image, dtype=tf.float32)
    # tf.debugging.assert_less_equal(raw_image, self.BIT_DEPTH)
    
    raw_image = tf.maximum(raw_image - self.BLACK_LEVEL, 0)
    raw_image /= (self.BIT_DEPTH - self.BLACK_LEVEL)
    raw_image.set_shape([None, None, 4])
    # raw_image = tf.image.resize_with_crop_or_pad(raw_image, img_height, img_width)
    # raw_image = (raw_image - 0.5) * 2  # scale from [0, 1] to [-1, 1]

    rgb_ground_truth = tf.image.decode_image(
        key_to_feature[MAI_RGB_GROUND_TRUTH.key]
    )
    rgb_ground_truth = tf.cast(rgb_ground_truth, tf.float32)
    rgb_ground_truth = self._normalize_rgb_image(rgb_ground_truth)
    rgb_ground_truth.set_shape([None, None, 3])

    large_feat_map_shape = tf.cast(key_to_feature[MAI_RGB_B5_3E_SHAPE.key], tf.int32)
    large_feat_map = tf.sparse.to_dense(key_to_feature[MAI_RGB_B5_3E.key])
    large_feat_map = tf.reshape(large_feat_map, large_feat_map_shape)
    large_feat_map = tf.squeeze(large_feat_map, axis=0)

    mid_feat_map_shape = tf.cast(key_to_feature[MAI_RGB_B5_5G_SHAPE.key], tf.int32)
    mid_feat_map = tf.sparse.to_dense(key_to_feature[MAI_RGB_B5_5G.key])
    mid_feat_map = tf.reshape(mid_feat_map, mid_feat_map_shape)
    mid_feat_map = tf.squeeze(mid_feat_map, axis=0)
    
    small_feat_map_shape = tf.cast(key_to_feature[MAI_RGB_B5_7C_SHAPE.key], tf.int32)
    small_feat_map = tf.sparse.to_dense(key_to_feature[MAI_RGB_B5_7C.key])
    small_feat_map = tf.reshape(small_feat_map, small_feat_map_shape)
    small_feat_map = tf.squeeze(small_feat_map, axis=0)

    return (
      {
        dataset_element.MAI_SMAPLE_ID: sample_id,
        dataset_element.MAI_RAW_PATCH: raw_image,
        dataset_element.MAI_DSLR_PATCH: rgb_ground_truth,
      },
      {
        model_prediction.ENHANCE_RGB: rgb_ground_truth,
        model_prediction.LARGE_FEAT: large_feat_map,
        model_prediction.MID_FEAT: mid_feat_map,
        model_prediction.SMALL_FEAT: small_feat_map,
      }
    )


if __name__ == '__main__':

  def check_b5():
    train_set = MaiIspB5TFRecordDataset(
        tf_record_path_pattern='/home/ron/Downloads/LearnedISP/tfrecord_b5/mai_isp.*.tfrecord'
      ).create_dataset(
          batch_size=32,
          num_readers=4,
          num_parallel_calls=8
      ).prefetch(tf.data.AUTOTUNE)

    train_sample = []
    for d, y in train_set:
      for k, v in y.items():
        print(k, v.shape)
      break    
  
  def check_split():
    train_set = MaiIspTFRecordDataset(
        tf_record_path_pattern='/home/ron/Downloads/LearnedISP/tfrecord/mai_isp.*.tfrecord'
      ).create_dataset(
          batch_size=64,
          num_readers=4,
          num_parallel_calls=8
      ).prefetch(tf.data.AUTOTUNE)

    val_set = MaiIspTFRecordDataset(
        tf_record_path_pattern='/home/ron/Downloads/LearnedISP/tfrecord/mai_isp_val.*.tfrecord'
      ).create_dataset(
          batch_size=64,
          num_readers=4,
          num_parallel_calls=8
      ).prefetch(tf.data.AUTOTUNE)

    train_sample = []
    for d, y in train_set:
      idx = list(d[dataset_element.MAI_SMAPLE_ID].numpy())
      train_sample += idx
    
    val_sample = []
    for d, y in val_set:
      idx = list(d[dataset_element.MAI_SMAPLE_ID].numpy())
      val_sample += idx
    
    train_sample = set(train_sample)
    val_sample = set(val_sample)
    union = train_sample.intersection(val_sample)
    print(min(train_sample), max(train_sample))
    print(min(val_sample), max(val_sample))
    import pdb; pdb.set_trace()
    print(union)
  
  def test_memory_leak():
    import os, psutil
    process = psutil.Process(os.getpid())

    print(f"Start: {process.memory_info().rss / 2**20: .2f}")  # in bytes 

    train_set = MaiIspTFRecordDataset(
        tf_record_path_pattern='/home/ron/Downloads/LearnedISP/tfrecord/mai_isp.*.tfrecord'
      ).create_dataset(
          batch_size=64,
          num_readers=4,
          num_parallel_calls=8
      ).repeat().prefetch(tf.data.AUTOTUNE)

    val_set = MaiIspTFRecordDataset(
        tf_record_path_pattern='/home/ron/Downloads/LearnedISP/tfrecord/mai_isp_val.*.tfrecord'
      ).create_dataset(
          batch_size=64,
          num_readers=4,
          num_parallel_calls=8
      ).prefetch(tf.data.AUTOTUNE)

    print(f"Dataset created: {process.memory_info().rss / 2**20: .2f}")  # in bytes 
    
    for j in range(10):
      for i, (x, y) in enumerate(train_set):
        if i % 100 == 0:
          print(f"Dataset Epoch[{j}] Iter[{i}]: {process.memory_info().rss / 2**20: .2f}")  # in bytes 
        if i > 20000:
          break

  with logger.catch():
    # mai_isp = MaiIspTFRecordDataset(
    #   tf_record_path_pattern='/home/ron/Downloads/LearnedISP/tfrecord/*.tfrecord')
    # dataset = mai_isp.create_dataset(batch_size=8)
    # for data in dataset:
    #   for d in data:
    #     for k, v in d.items():
    #       print(k, v.shape, v.dtype)
    #   break

    # check_split()
    # test_memory_leak()
    check_b5()
    

  
