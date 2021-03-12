import os
import glob

from imageio.core.util import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import fire
import imageio
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from PIL import Image
from loguru import logger
from tensorflow.keras import callbacks

from isp import metrics
from isp import losses
from isp import callbacks
from isp import experiment
from isp.model import io
from isp.data import dataset
from isp.model.unet import UNet, UNetGrid, UNetResX2R, UNetRes
from isp.model import layers
from isp.model import unet


TEST_DIR = "/home/ron/Downloads/LearnedISP/MAI2021_LearnedISP_valid_raw"


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

  unet = UNetRes('train', alpha=0.25)
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
    callbacks=[
      # tensorbaord,
      # checkpoint,
      # write_image,
    ]
  )


def run_interpreter(inter, x):
  input_tensor_id = inter.get_input_details()[0]['index']
  inter.set_tensor(input_tensor_id, x)
  inter.invoke()

  output_tensor_id = inter.get_output_details()[0]['index']
  lite_pred = inter.get_tensor(output_tensor_id)
  return lite_pred


def tflite_convert(model_dir, output_path, in_size=[544, 960]):

  """
  take the savedmodel and converted it with CLI tool from tf v1.5:
  tflite_convert --output_file tf15.lite \
    --saved_model_dir checkpoints/unet_05_grid/saved \
    --output_arrays "StatefulPartitionedCall/StatefulPartitionedCall/model/tf.nn.depth_to_space/DepthToSpace"
  """

  def remove_weight_norm(model: tf.keras.Model):
    for layer in model.layers:
      if isinstance(layer, layers.WeightNormalization):
        layer.remove()
      elif hasattr(layer, 'layers'):
        remove_weight_norm(layer)
  
  net = UNet('export', alpha=1.0)
  payload = np.random.normal(size=[1, *in_size, 4]).astype(np.float32)
  net.predict({
    io.dataset_element.MAI_RAW_PATCH: payload
  })
  # net.load_weights(model_dir)
  
  tf_pred = net.predict({
    io.dataset_element.MAI_RAW_PATCH: payload
  })
  # remove_weight_norm(net)
  
  x = tf.keras.Input(shape=[*in_size, 4], batch_size=1, dtype=tf.float32)
  y = net._call(x)
  # y = tf.cast(tf.clip_by_value(y, 0.0, 1.0) * 255, tf.uint8)
  functional_net = tf.keras.models.Model(x, y)
  functional_net.predict(np.zeros([1, *in_size, 4]))
  # import pdb; pdb.set_trace()
  tf.saved_model.save(functional_net, os.path.join(model_dir, 'saved'))
  # for z in functional_net.inputs:
  #   z.set_shape([320, 240, 4])
  functional_net.summary()
  
  converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(model_dir, 'saved'))
  # converter = tf.lite.TFLiteConverter.from_keras_model(functional_net)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()

  with open(output_path, mode='wb') as f:
    f.write(tflite_model)
  
  """
  Test run
  """
  
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


def run_two_stage_experiment(config_path, load_weight=None, skip_stage_1=False):
  config = experiment.ExperimentConfig(config_path)
  exp = experiment.TwoStageExperiment(config)
  exp.train(load_weight=load_weight, skip_stage_1=skip_stage_1)


def run_3_stage_experiment(config_path, load_weight=None, skip_stage_1=False, skip_stage_2=False):
  config = experiment.ExperimentConfig(config_path)
  exp = experiment.ThreeStageExperiment(config)
  exp.train(load_weight=load_weight, skip_stage_1=skip_stage_1, skip_stage_2=skip_stage_2)


def run_ctx_experiment(config_path, load_weight=None, quantize=False):
  logger.info(f"[run_ctx_experiment] config_path: {config_path}")
  logger.info(f"[run_ctx_experiment] load_weight: {load_weight}")
  logger.info(f"[run_ctx_experiment] quantize: {quantize}")

  config = experiment.ExperimentConfig(config_path)
  exp = experiment.CtxLossExperiment(config)
  exp.train(load_weight=load_weight, quantize=quantize)


def test_cobi():
  import torch
  from cobi_torch import contextual_bilateral_loss
  A = tf.random.uniform([2, 8, 8, 32], minval=-1.0, maxval=1.0)
  B = tf.random.uniform([2, 8, 8, 32], minval=-1.0, maxval=1.0)

  patch_size = 3
  cobi = losses.CoBi(patch_size=[patch_size, patch_size])

  _A = tf.image.extract_patches(
    A,
    [1, patch_size, patch_size, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    'VALID')
  _B = tf.image.extract_patches(
    B,
    [1, patch_size, patch_size, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    'VALID')
  
  with torch.no_grad():
    C2 = contextual_bilateral_loss(
      torch.tensor(_A.numpy()).permute(0, 3, 1, 2),
      torch.tensor(_B.numpy()).permute(0, 3, 1, 2),
      loss_type='cosine'
    )
    C2 = C2.cpu().numpy()
    C1 = cobi(A, B)
    C1 = C1.numpy()
    d = np.abs(C1 - C2)
    print(d.max())


def test_model():
  net = unet.UNetCoBi('train')
  input_dict = {
    io.dataset_element.MAI_RAW_PATCH:
      np.ones([1, 128, 128, 4], dtype=np.float32),
    io.model_prediction.INTER_MID_PRED:
      np.ones([1, 256, 256, 3], dtype=np.float32),
  }
  net.predict(input_dict)
  net.summary()


def predict_test_set(config_path, weight_path, output_dir, test_dir=TEST_DIR, device='/gpu:0'):
  """
  python3 main.py predict_test_set configs/unet_05_ctx.json  \
    checkpoints/unet_05_ctx_0305/checkpoint \
    /home/ron/Downloads/LearnedISP/eval_out
  
  runtime per image [s] : 1.0 
  CPU[1] / GPU[0] : 1
  Extra Data [1] / No Extra Data [0] : 1
  Other description : Solution
  """
  with tf.device(device):
    config = experiment.ExperimentConfig(config_path)
    config.model['init_param']['args'] = ['eval']
    builder = experiment.ExperimentBuilder(config)
    model = builder.compilted_model()
    model.load_weights(weight_path)

    os.makedirs(output_dir, exist_ok=True)
    raw_imgs = glob.glob(os.path.join(test_dir, '*.png'))
    logger.info(f'Find {len(raw_imgs)} eval raw images')

    batched = []
    batched_idx = []
    B = 32
    raw_val_max = -1
    raw_val_min = 2**20
    for i in range(0, len(raw_imgs), B):
      print(f"{i} => {i + B}")
      for j in range(i, min(i + B, len(raw_imgs))):
        I = np.asarray(imageio.imread(raw_imgs[j]))
        I = I.astype(np.float32) - dataset.MaiIspTFRecordDataset.BLACK_LEVEL
        I /= dataset.MaiIspTFRecordDataset.BIT_DEPTH - dataset.MaiIspTFRecordDataset.BLACK_LEVEL
        raw_val_max = max(I.max(), raw_val_max)
        raw_val_min = min(I.min(), raw_val_min)
        batched.append(I)
        batched_idx.append(j)    
      
      batched = np.stack(batched)[..., None]
      tf_raws = tf.nn.space_to_depth(batched, 2)

      preds = model.predict({
        io.dataset_element.MAI_RAW_PATCH: tf_raws,
      })
      preds = preds[io.model_prediction.ENHANCE_RGB]
      preds = tf.clip_by_value(preds, 0, 1)
      preds = tf.cast(preds * 255, tf.uint8).numpy()
      
      for pred, img_id in zip(preds, batched_idx):
        src_path = raw_imgs[img_id]
        src_name = os.path.basename(src_path)  # *.png
        dst_path = os.path.join(output_dir, src_name)
        Image.fromarray(pred).save(dst_path)
      batched = []
      batched_idx = []
    
    print(f"raw_val_max: {raw_val_max}")
    print(f"raw_val_min: {raw_val_min}")


def export_pb(frozen_out_path, in_size=[544, 960]):
  from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_configs
  NoOpQuantizeConfig = default_8bit_quantize_configs.NoOpQuantizeConfig

  def quantize(functional_net):
    def apply_quantization_to_dense(layer):
      no_quan_layers = (
        tf.keras.layers.UpSampling2D,
        tf.keras.layers.Concatenate,
        tf.keras.layers.Lambda)
      if isinstance(layer, no_quan_layers):
        return tfmot.quantization.keras.quantize_annotate_layer(
          layer, quantize_config=NoOpQuantizeConfig())
      else:
        return tfmot.quantization.keras.quantize_annotate_layer(layer)

    # Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense` 
    # to the layers of the model.
    annotated_model = tf.keras.models.clone_model(
        functional_net,
        clone_function=apply_quantization_to_dense,
    )

    # quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    with tf.keras.utils.custom_object_scope({"NoOpQuantizeConfig": NoOpQuantizeConfig}):
      quant_aware_model = tfmot.quantization.keras.quantize_model(annotated_model)
    quant_aware_model.summary()
    return quant_aware_model

  def get_keras_model():
    net = UNet('export', alpha=1.0)
    payload = np.random.normal(size=[1, *in_size, 4]).astype(np.float32)
    net.predict({
      io.dataset_element.MAI_RAW_PATCH: payload
    })
    # net.load_weights(model_dir)
    
    tf_pred = net.predict({
      io.dataset_element.MAI_RAW_PATCH: payload
    })
    # remove_weight_norm(net)
    
    x = tf.keras.Input(shape=[*in_size, 4], batch_size=1, dtype=tf.float32)
    y = net._call(x)
    # y = tf.cast(tf.clip_by_value(y, 0.0, 1.0) * 255, tf.uint8)
    functional_net = tf.keras.models.Model(x, y)
    functional_net.predict(np.zeros([1, *in_size, 4]))
    return functional_net

  from tensorflow import keras
  from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
  #path of the directory where you want to save your model
  # name of the .pb file
  frozen_graph_filename = "frozen_graph"
  model = get_keras_model()
  # model = quantize(model)
  
  # Convert Keras model to ConcreteFunction
  full_model = tf.function(lambda x: model(x))
  full_model = full_model.get_concrete_function(
      tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
  # Get frozen ConcreteFunction
  # frozen_func = convert_variables_to_constants_v2(full_model)
  frozen_func = full_model
  frozen_func.graph.as_graph_def()
  layers = [op.name for op in frozen_func.graph.get_operations()]
  print("-" * 60)
  print("Frozen model layers: ")
  for layer in layers:
      print(layer)
  print("-" * 60)
  print("Frozen model inputs: ")
  print(frozen_func.inputs)
  print("Frozen model outputs: ")
  print(frozen_func.outputs)
  # Save frozen graph to disk
  tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=frozen_out_path,
                    name=f"{frozen_graph_filename}.pb",
                    as_text=False)
  # Save its text representation
  tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=frozen_out_path,
                    name=f"{frozen_graph_filename}.pbtxt",
                    as_text=True)
  
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()
  with open("export_pb.tflite", mode='wb') as f:
    f.write(tflite_model)


def test_save_h5(in_size=[256, 256]):
  
  def get_keras_model():
    net = UNet('export', alpha=1.0)
    payload = np.random.normal(size=[1, *in_size, 4]).astype(np.float32)
    net.predict({
      io.dataset_element.MAI_RAW_PATCH: payload
    })
    # net.load_weights(model_dir)
    
    tf_pred = net.predict({
      io.dataset_element.MAI_RAW_PATCH: payload
    })
    # remove_weight_norm(net)
    
    x = tf.keras.Input(
      shape=[*in_size, 4], batch_size=1, dtype=tf.float32,
      name=io.dataset_element.MAI_RAW_PATCH)
    y = net._call(x)
    # y = tf.cast(tf.clip_by_value(y, 0.0, 1.0) * 255, tf.uint8)
    functional_net = tf.keras.models.Model(x, y)
    functional_net.predict(np.zeros([1, *in_size, 4]))
    return net, functional_net

  _, model = get_keras_model()
  tf.keras.models.save_model(model, 'test_h5.h5', save_format='h5')

  payload = np.ones([1, *in_size, 4]).astype(np.float32)
  pred = model.predict({
    io.dataset_element.MAI_RAW_PATCH: payload
  })
  print(pred.mean(), pred.shape)


if __name__ == '__main__':
  soft_gpu_meme_growth()

  with logger.catch(reraise=True):
    with tf.device('/gpu:0'):
      os.system("nvidia-settings -a '[gpu:0]/GPUPowerMizerMode=1'")  # make sure GPU is using maximument performance mode

      fire.Fire({
        'simple_train': simple_train,
        'tflite_convert': tflite_convert,
        'eval_tflite_model': eval_tflite_model,
        'eval_tf_model': eval_tf_model,
        'run_experiment': run_experiment,
        'run_two_stage_experiment': run_two_stage_experiment,
        'run_3_stage_experiment': run_3_stage_experiment,
        'run_ctx_experiment': run_ctx_experiment,
        'test_cobi': test_cobi,
        'test_model': test_model,
        'predict_test_set': predict_test_set,
        'export_pb': export_pb,
        'test_save_h5': test_save_h5,
      })
      # simple_train('./checkpoints/unet_res_bil_hyp_large')

      # board_check()