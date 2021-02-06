import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
from loguru import logger

from isp.model.unet import UNet
from isp.model import io
from isp.data import dataset
from isp import metrics
from isp import losses


def soft_gpu_meme_growth():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)


def iter_data(dataset):
  for i, d in enumerate(dataset):
    if i % 200:
      print(i)


def simple_train():

  unet = UNet('train', alpha=0.5)
  psnr = metrics.PSNR(
    io.dataset_element.MAI_DSLR_PATCH,
    io.model_prediction.ENHANCE_RGB
  )
  ms_ssim = losses.MSSSIM(
    # io.dataset_element.MAI_DSLR_PATCH,
    # io.model_prediction.ENHANCE_RGB
  )
  mse = tf.keras.losses.MeanSquaredError()
  adam = tf.keras.optimizers.Adam(learning_rate=1e-4)

  mai_isp = dataset.MaiIspTFRecordDataset(
    tf_record_path_pattern='/home/ron/Downloads/LearnedISP/tfrecord/*.tfrecord')
  train_set, val_set = dataset.split_train_valid(
    mai_isp.create_dataset(
      batch_size=128,
      num_readers=4,
      num_parallel_calls=8),
    val_ratio_or_size=10)
  
  # iter_data(train_set)
  iter_data(val_set)
  
  unet.compile(
    optimizer=adam,
    loss={
      io.model_prediction.ENHANCE_RGB: mse
    },
    metrics={
      io.model_prediction.ENHANCE_RGB: psnr,
    }
  )
  # unet.summary()

  unet.fit(
    train_set,
    steps_per_epoch=4000,
    epochs=10,
    validation_data=val_set,
    use_multiprocessing=True,
    workers=4,
  )


with logger.catch():
  soft_gpu_meme_growth()
  simple_train()