import os

from tensorflow.keras import callbacks

from isp.model import unet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
from loguru import logger

from isp.model.unet import UNet, UNetResX2R, UNetRes
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


def iter_data(dataset, max_iter=100):
  for i, d in enumerate(dataset):
    if i % 200:
      print(i)
    if i > max_iter:
      break


def sanity_check(model: tf.keras.models.Model, dataset: tf.data.Dataset):
  for x, y in dataset:
    pred = model.predict(x)
    np_label = y[io.model_prediction.ENHANCE_RGB].numpy()
    print(np_label.max(), np_label.min())
    
    np_pred = pred[io.model_prediction.ENHANCE_RGB]
    print(np_pred.max(), np_pred.min())
    # import pdb; pdb.set_trace()
    break
  model.summary()


def board_check():
  import pkg_resources

  for entry_point in pkg_resources.iter_entry_points('tensorboard_plugins'):
      print(entry_point.dist)


def simple_train(model_dir):
  # policy = tf.keras.mixed_precision.Policy('mixed_float16')
  # tf.keras.mixed_precision.set_global_policy(policy)

  unet = UNetRes('train', alpha=0.5)
  psnr = metrics.PSNR(
    io.dataset_element.MAI_DSLR_PATCH,
    io.model_prediction.ENHANCE_RGB
  )
  ms_ssim = losses.HypbirdSSIM(
    # io.dataset_element.MAI_DSLR_PATCH,
    # io.model_prediction.ENHANCE_RGB
  )
  mse = tf.keras.losses.MeanSquaredError()
  adam = tf.keras.optimizers.Adam(learning_rate=1e-4)

  train_set = dataset.MaiIspTFRecordDataset(
      tf_record_path_pattern='/home/ron/Downloads/LearnedISP/tfrecord/mai_isp.*.tfrecord'
    ).create_dataset(
        batch_size=64,
        num_readers=4,
        num_parallel_calls=8
    ).repeat().prefetch(tf.data.AUTOTUNE)

  val_set = dataset.MaiIspTFRecordDataset(
      tf_record_path_pattern='/home/ron/Downloads/LearnedISP/tfrecord/mai_isp_val.*.tfrecord'
    ).create_dataset(
        batch_size=64,
        num_readers=4,
        num_parallel_calls=8
    ).prefetch(tf.data.AUTOTUNE)
    
  
  # iter_data(train_set)
  # iter_data(val_set)
  # unet.load_weights(model_dir)
  sanity_check(unet, val_set)
  
  unet.compile(
    optimizer=adam,
    loss={
      io.model_prediction.ENHANCE_RGB: ms_ssim
    },
    metrics={
      io.model_prediction.ENHANCE_RGB: psnr,
    }
  )
  # unet.summary()

  os.makedirs(model_dir, exist_ok=True)
  log_dir = os.path.join(model_dir, 'logs')
  tensorbaord = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    write_images=True,
    write_graph=True)
  
  ckpt_path = os.path.join(model_dir, 'checkpoint')
  checkpoint = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    monitor=psnr.name,
    save_best_only=True,
  )

  unet.fit(
    train_set,
    steps_per_epoch=2000,
    epochs=20,
    validation_data=val_set,
    use_multiprocessing=True,
    workers=4,
    callbacks=[
      tensorbaord,
      checkpoint,
    ]
  )


with logger.catch():
  soft_gpu_meme_growth()
  simple_train('./checkpoints/unet_res_bil_hyp')

  # board_check()