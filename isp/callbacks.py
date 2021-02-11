import numpy as np
import tensorflow as tf


class SaveValImage(tf.keras.callbacks.Callback):
  """
  Must using with CacheOutput metric to passthrough model's output to 'logs' args
  """

  def __init__(self,
              log_dir,
              train_summary_freq=200,
              test_summary_freq=2,
              sample_per_batch=4,
              log_key='enhanced_rgb_cache_output'):
    super().__init__()
    self.writer = tf.summary.create_file_writer(log_dir)
    self.train_summary_freq = train_summary_freq
    self.test_summary_freq = test_summary_freq
    self.sample_per_batch = sample_per_batch
    self.log_key = log_key

    self.train_steps_cnt = 0
    self.test_steps_cnt = 0
  
  def fp_img_to_uint8(self, fp_img):
    img = np.clip(fp_img * 255, 0, 255)
    img = img.astype(np.uint8)
    return img
  
  def on_batch_end(self, batch, logs):
    self.train_steps_cnt += 1
    
  def on_test_batch_end(self, batch, logs):
    self.test_steps_cnt += 1
    global_step = self.train_steps_cnt + self.test_steps_cnt
    
    if self.test_steps_cnt % self.test_summary_freq == 0:
      # import pdb; pdb.set_trace()
      match = list(filter(lambda x: self.log_key in x, logs.keys()))
      assert len(match) == 1
      last_eval = logs[match[0]]

      with self.writer.as_default():
        pred_img = self.fp_img_to_uint8(last_eval[0])
        true_img = self.fp_img_to_uint8(last_eval[1])
        tf.summary.image(
          'predict_images',
          pred_img[:self.sample_per_batch],
          step=global_step
        )
        tf.summary.image(
          'groundtruth_images',
          true_img[:self.sample_per_batch],
          step=global_step,
        )


class DevCall(tf.keras.callbacks.Callback):
  """
  Must using with CacheOutput metric to passthrough model's output to 'logs' args
  """

  def __init__(self):
    super().__init__()
    self.train_steps_cnt = 0
    self.test_steps_cnt = 0
  
  def on_batch_end(self, batch, logs):
    print('batch: ', self.train_steps_cnt)
    print('-' * 100)
    self.train_steps_cnt += 1
  
    
  def on_test_batch_end(self, batch, logs):
    print('test batch: ', self.test_steps_cnt)
    print('-' * 100)
    self.test_steps_cnt += 1