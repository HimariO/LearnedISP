# %%
import os

from tensorflow.python.keras.backend import dtype
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import fire
import torch
import numpy as np
import tensorflow as tf
from PIL import Image
from loguru import logger
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

from isp.model import io
from isp import metrics
from isp import losses
from isp.data import dataset
from isp.model.unet import UNet, UNetResX2R, UNetRes, UNetGrid
from isp.model import unet
from isp.model import layers
from isp import experiment


def remove_weight_norm(model: tf.keras.Model):
    for layer in model.layers:
      if isinstance(layer, layers.WeightNormalization):
        layer.remove()
      elif hasattr(layer, 'layers'):
        remove_weight_norm(layer)

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

soft_gpu_meme_growth()

# %%

net = UNetRes('train', alpha=0.5)
net.predict({
    io.dataset_element.MAI_RAW_PATCH: 
      np.zeros([1, 128, 128, 4], dtype=np.float32)
})
# net.load_weights('checkpoints/dev/checkpoint')
net.summary()

in_layer = tf.keras.Input([128, 128, 4])
y = net._call(in_layer)
fun_net = tf.keras.Model(in_layer, y)

# %%

net2 = UNetRes('train', alpha=0.5, weight_norm=False)
t = net2.predict({
    io.dataset_element.MAI_RAW_PATCH: 
      np.zeros([1, 128, 128, 4], dtype=np.float32)
})
net2.load_weights('checkpoints/dev/checkpoint')
remove_weight_norm(net2)
# net2.build([1, 320, 240, 4])
# %%

net3 = UNetRes('train', alpha=0.5)
x = tf.keras.Input(shape=[320, 240, 4], batch_size=1, dtype=tf.float32)
y = net3._call(x)
functional_net = tf.keras.models.Model(x, y)

# %%

lite_net = tf.lite.Interpreter(model_path='net.tflite')

# %%

val_set = dataset.MaiIspTFRecordDataset(
      tf_record_path_pattern='/home/ron/Downloads/LearnedISP/tfrecord/mai_isp_val.*.tfrecord'
    ).create_dataset(
        batch_size=64,
        num_readers=4,
        num_parallel_calls=8
    )

# %%

for x, y in val_set:
  t = net.predict(x)
  break
# %%


idx = 3
img = np.clip(t['enhanced_rgb'][idx], 0, 1) * 255
img = img.astype(np.uint8)
plt.imshow(img)
plt.title('prediction')
plt.show()

yimg = np.clip(y['enhanced_rgb'][idx], 0, 1) * 255
yimg = yimg.astype(np.uint8)
plt.imshow(yimg)
plt.title('ground truth')
plt.show()
# %%

def run_interpreter(inter, x):
  input_tensor_id = inter.get_input_details()[0]['index']
  inter.set_tensor(input_tensor_id, x)
  inter.invoke()
  
  output_tensor_id = inter.get_output_details()[0]['index']
  lite_pred = inter.get_tensor(output_tensor_id)
  return lite_pred

lite_net = tf.lite.Interpreter(model_path='net.tflite')
lite_net.allocate_tensors()

lite_pred = run_interpreter(lite_net, x['mai2021_raw_img_patch'].numpy()[idx: idx+1])
# %%

img2 = np.clip(lite_pred[0], 0, 1) * 255
img2 = img.astype(np.uint8)
plt.imshow(img2)

# %%

psnr = metrics.PSNR(
  io.dataset_element.MAI_DSLR_PATCH,
  io.model_prediction.ENHANCE_RGB
)

A = np.zeros([1, 20, 20, 3], dtype=np.float32)
B = np.ones([1, 20, 20, 3], dtype=np.float32)
p = tf.image.psnr(A, B, 1.0)
print(p)
# %%

config_path = 'configs/unet_05_hypbrid_loss.json'
config = experiment.ExperimentConfig(config_path)
exp = experiment.Experiment(config)
tf_dataset = exp.builder.get_train_dataset()

# %%

for x, y in tf_dataset:
  raw_rgb = x[io.dataset_element.MAI_RAW_PATCH][0, ..., :3]
  plt.imshow(raw_rgb)
  plt.show()

  dslr = y[io.model_prediction.ENHANCE_RGB][0]
  plt.imshow(dslr)
  plt.show()

  break
# %%

import os, psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss // 2**20)  # in bytes 


# %%

A = tf.fill([1, 4, 4, 1], 0)
B = tf.fill([1, 4, 4, 1], 64)
C = tf.fill([1, 4, 4, 1], 128)
D = tf.fill([1, 4, 4, 1], 255)

A = tf.cast(A, tf.uint8)
B = tf.cast(B, tf.uint8)
C = tf.cast(C, tf.uint8)
D = tf.cast(D, tf.uint8)

# bayer = tf.stack([A, B, C, D], axis=-1)
# %%

Pa = tf.pad(B, [(0, 0), (0, 0), (0, 0), (0, 3)], constant_values=0)
Ta = tf.nn.depth_to_space(Pa, 2)
plt.imshow(Ta[0])
plt.show()

Taaa = tf.nn.depth_to_space(tf.concat([Pa, Pa, Pa], -1), 2)
plt.imshow(Taaa[0])
plt.show()
# %%

Pb = tf.pad(B, [(0, 0), (0, 0), (0, 0), (1, 2)], constant_values=0)
Tb = tf.nn.depth_to_space(Pb, 2)
plt.imshow(Tb[0])
# %%

bay_unet = unet.functinoal_unet_bay()
bay_unet.summary()
# %%

tf.image.rgb_to_hsv(tf.cast(Pb[..., :3], tf.float32))
# %%


config_path = './configs/unet_05_curl.json'
config = experiment.ExperimentConfig(config_path)
exp = experiment.Experiment(config)

val_set = exp.builder.get_val_dataset()
net = exp.builder.compilted_model()
net.load_weights('checkpoints/unet_05_curl_plu/checkpoint')
# %%

for x, y in val_set:
  raw_rgb = tf.stack([
    x[io.dataset_element.MAI_RAW_PATCH][0, ..., 0] / 1,
    tf.reduce_mean(x[io.dataset_element.MAI_RAW_PATCH][0, ..., 1:3], axis=-1) / 1,
    x[io.dataset_element.MAI_RAW_PATCH][0, ..., 3] / 1,
  ], axis=-1)
  plt.imshow(raw_rgb)
  plt.show()
  
  plt.imshow(tf.nn.depth_to_space(x[io.dataset_element.MAI_RAW_PATCH], 2)[0, ..., 0])
  plt.show()

  plt.imshow(raw_rgb)
  plt.show()

  dslr = y[io.model_prediction.ENHANCE_RGB][0]
  plt.imshow(dslr)
  plt.show()
  
  # x[io.dataset_element.MAI_RAW_PATCH] *= 4

  p = net(x)
  p['rgb_color_curve'] = tf.nn.relu(p['rgb_color_curve'])
  p['rgb_color_curve'] = tf.reshape(p['rgb_color_curve'], [-1, 3, 16])
  
  pred_img = tf.clip_by_value(p[io.model_prediction.INTER_MID_PRED][0], 0, 1)
  print("INTER_MID_PRED: ", tf.image.psnr(pred_img, dslr, 1.0))
  plt.imshow(pred_img)
  plt.show()

  adj_pred_img = tf.clip_by_value(p[io.model_prediction.ENHANCE_RGB][0], 0, 1)
  print("ENHANCE_RGB: ", tf.image.psnr(adj_pred_img, dslr, 1.0))
  plt.imshow(adj_pred_img)
  plt.show()

  delta = tf.abs(pred_img - adj_pred_img)
  delta = tf.reduce_mean(delta, axis=-1)
  plt.imshow(delta)
  plt.colorbar()
  plt.show()
  break
# %%

with tf.device('/gpu:1'):
  conv_a = tf.keras.layers.Conv2D(8, 3)
  conv_b = tf.keras.layers.Conv2D(8, 3)
  print(conv_a.get_weights())

  data = np.ones([1, 16, 16, 3])
  da = conv_a(data)
  db = conv_b(data)
  print(np.abs(da -  db).mean())

  conv_b.kernel = conv_a.kernel
  da = conv_a(data)
  db = conv_b(data)
  print(np.abs(da -  db).mean())

  I = tf.keras.layers.Input(shape=[16, 16, 3])
  conv_c = tf.keras.layers.Conv2D(8, 3)
  conv_c(I)
  print(conv_c.variables)

# %%

class Module(tf.keras.Model):
  M_C = 0

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    M_C = Module.M_C
    self.fc1 = tf.keras.layers.Dense(10, name=f"moduel_{M_C}_fc1")
    self.add = tf.keras.layers.Add(name=f"moduel_{M_C}_add")
    Module.M_C += 1 
  
  def call(self, x):
    return self._call(x)
  
  def _call(self, x):
    return self.add([self.fc1(x), x])

class Base(tf.keras.Model):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.fc1 = tf.keras.layers.Dense(10)
    self.m = Module()
  
  def call(self, x):
    return self._call(x)
  
  def _call(self, x):
    h = self.fc1(x)
    h = self.m(h)
    return h
  
  def _call2(self, x):
    h = self.fc1(x)
    h = self.m._call(h)
    return h

Module()
Module()
base = Base()

I = tf.keras.layers.Input([16, 16, 3])
Y = base._call2(I)
base_faltten = tf.keras.Model(I, Y)

base_faltten.summary()
# base.summary()
# %%

net = unet.functional_unet_grid(mode='functional', alpha=0.5, batch_size=1)
net.summary()

net.predict(np.ones([1, 128, 128,4]))

lite_conv = tf.lite.TFLiteConverter.from_keras_model(net)
with open('check_func.tflite', mode='wb') as f:
  f.write(lite_conv.convert())
# %%


with tf.device('/gpu:1'):
  img_path = "/home/ron/Pictures/dog.jpeg"
  dog_img = np.asarray(Image.open(img_path).resize((331, 331))).astype(np.float32)[None, ...]
  dog_img = tf.keras.applications.nasnet.preprocess_input(dog_img)
  B5 = tf.keras.applications.NASNetLarge()
  # B5 = EfficientNetB5(input_shape=[256, 256, 3], include_top=False)
  # B5.summary()
  # pred = B5.predict(np.ones([1, 256, 256, 3]))
  pred = B5.predict(dog_img)
  print(pred[0, :100])
  print(pred.argmax(), pred.max())
  print(pred.mean())
# %%
