import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import fire
import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.keras import callbacks

from isp import metrics
from isp import losses
from isp import callbacks
from isp import experiment
from isp.model import io
from isp.data import dataset
from isp.model.unet import UNet, UNetResX2R, UNetRes
from isp.model import layers


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


def simple_train(model_dir, load_weight=None):
  # policy = tf.keras.mixed_precision.Policy('mixed_float16')
  # tf.keras.mixed_precision.set_global_policy(policy)

  unet = UNet('train', alpha=1.5)
  psnr = metrics.PSNR()
  cache_model_out = metrics.CacheOutput()
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
    )
    
  
  # iter_data(train_set)
  # iter_data(val_set)
  # unet.load_weights(model_dir)
  sanity_check(unet, val_set)
  
  unet.compile(
    optimizer=adam,
    loss={
      io.model_prediction.ENHANCE_RGB: mse
    },
    metrics={
      io.model_prediction.ENHANCE_RGB: [psnr, cache_model_out],
    }
  )
  # unet.summary()

  if load_weight is not None:
    unet.load_weights(load_weight)

  os.makedirs(model_dir, exist_ok=True)
  log_dir = os.path.join(model_dir, 'logs')
  tensorbaord = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    write_images=True,
    write_graph=True)
  write_image = callbacks.SaveValImage(log_dir)
  
  ckpt_path = os.path.join(model_dir, 'checkpoint')
  checkpoint = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    monitor=psnr.name,
    save_best_only=False,
  )

  unet.fit(
    train_set,
    steps_per_epoch=2000,
    epochs=20,
    validation_data=val_set,
    use_multiprocessing=False,
    workers=1,
    callbacks=[
      tensorbaord,
      checkpoint,
      write_image,
    ]
  )


def run_interpreter(inter, x):
  input_tensor_id = inter.get_input_details()[0]['index']
  inter.set_tensor(input_tensor_id, x)
  inter.invoke()

  output_tensor_id = inter.get_output_details()[0]['index']
  lite_pred = inter.get_tensor(output_tensor_id)
  return lite_pred


def tflite_convert(model_dir, output_path):

  def remove_weight_norm(model: tf.keras.Model):
    for layer in model.layers:
      if isinstance(layer, layers.WeightNormalization):
        layer.remove()
      elif hasattr(layer, 'layers'):
        remove_weight_norm(layer)
  
  net = UNetRes('train', alpha=0.5, weight_norm=False)
  payload = np.random.normal(size=[1, 128, 128, 4]).astype(np.float32)
  net.predict({
    io.dataset_element.MAI_RAW_PATCH: payload
  })
  net.load_weights(model_dir)
  
  tf_pred = net.predict({
    io.dataset_element.MAI_RAW_PATCH: payload
  })
  remove_weight_norm(net)
  
  x = tf.keras.Input(shape=[128, 128, 4], batch_size=1, dtype=tf.float32)
  y = net._call(x)
  functional_net = tf.keras.models.Model(x, y)
  # for z in functional_net.inputs:
  #   z.set_shape([320, 240, 4])
  
  converter = tf.lite.TFLiteConverter.from_keras_model(functional_net)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()

  with open(output_path, mode='wb') as f:
    f.write(tflite_model)
  
  lite_net = tf.lite.Interpreter(model_path=output_path)
  lite_net.allocate_tensors()
  
  lite_pred = run_interpreter(lite_net, payload)

  print(np.abs(tf_pred[io.model_prediction.ENHANCE_RGB] - lite_pred).mean())


def eval_tflite_model(model_path):
  lite_net = tf.lite.Interpreter(model_path=model_path, num_threads=16)
  lite_net.allocate_tensors()

  val_set = dataset.MaiIspTFRecordDataset(
      tf_record_path_pattern='/home/ron/Downloads/LearnedISP/tfrecord/mai_isp_val.*.tfrecord'
    ).create_dataset(
        batch_size=64,
        num_readers=4,
        num_parallel_calls=8
    )
  
  psnr_all = []
  for i, (x, y) in enumerate(val_set):
    print(f'batch [{i}]')
    raw_patchs = x[io.dataset_element.MAI_RAW_PATCH].numpy()
    targets = y[io.model_prediction.ENHANCE_RGB].numpy()

    for patch, target in zip(raw_patchs, targets):
      lite_pred = run_interpreter(lite_net, patch[None, ...])
      lite_pred = np.clip(lite_pred, 0, 1.0)
      psnr_all.append(tf.image.psnr(lite_pred[0], target, 1.0).numpy())
      # batch_pred.append(lite_pred)
  mean_psnr = np.stack(psnr_all).mean()
  print(f"mean_psnr: {mean_psnr}")


def eval_tf_model(model_path):
  net = UNetRes('train', alpha=0.5)
  net.predict({
      io.dataset_element.MAI_RAW_PATCH: 
        np.zeros([1, 128, 128, 4], dtype=np.float32)
  })
  net.load_weights(model_path)

  psnr = metrics.PSNR(
    io.dataset_element.MAI_DSLR_PATCH,
    io.model_prediction.ENHANCE_RGB
  )
  cnt = metrics.Count()
  mse = tf.keras.losses.MeanSquaredError()
  net.compile(
    loss={
      io.model_prediction.ENHANCE_RGB: mse
    },
    metrics={
      io.model_prediction.ENHANCE_RGB: [psnr, cnt],
    }
  )

  val_set = dataset.MaiIspTFRecordDataset(
      tf_record_path_pattern='/home/ron/Downloads/LearnedISP/tfrecord/mai_isp_val.*.tfrecord'
    ).create_dataset(
        batch_size=64,
        num_readers=1,
        num_parallel_calls=8
    )
  
  psnr.reset_states()
  net.evaluate(val_set.take(1), callbacks=[callbacks.DevCall()])
  import pdb; pdb.set_trace()
  
  psnr_all = []
  for i, (x, y) in enumerate(val_set):
    print(f'batch [{i}]')
    targets = y[io.model_prediction.ENHANCE_RGB]
    predicts = net.predict(x)[io.model_prediction.ENHANCE_RGB]

    predicts = tf.clip_by_value(predicts, 0.0, 1.0)
    psnr_all.append(tf.image.psnr(predicts, targets, 1.0).numpy())
  mean_psnr = np.stack(psnr_all).mean()

  import pdb; pdb.set_trace()
  print(f"mean_psnr: {mean_psnr}")


def run_experiment(config_path, load_weight=None):
  config = experiment.ExperimentConfig(config_path)
  exp = experiment.Experiment(config)
  exp.train(load_weight=load_weight)


if __name__ == '__main__':

  with logger.catch():
    soft_gpu_meme_growth()

    fire.Fire({
      'simple_train': simple_train,
      'tflite_convert': tflite_convert,
      'eval_tflite_model': eval_tflite_model,
      'eval_tf_model': eval_tf_model,
      'run_experiment': run_experiment,
    })
    # simple_train('./checkpoints/unet_res_bil_hyp_large')

    # board_check()