# %%
import os

from tensorflow.python.keras.backend import dtype
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import fire
import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

from isp.model import io
from isp import metrics
from isp import losses
from isp.data import dataset
from isp.model.unet import UNet, UNetResX2R, UNetRes
from isp.model import layers
from isp import experiment


def remove_weight_norm(model: tf.keras.Model):
    for layer in model.layers:
      if isinstance(layer, layers.WeightNormalization):
        layer.remove()
      elif hasattr(layer, 'layers'):
        remove_weight_norm(layer)


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
