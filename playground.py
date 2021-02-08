# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import fire
import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.keras import callbacks

from isp.model import io
from isp import metrics
from isp import losses
from isp.data import dataset
from isp.model.unet import UNet, UNetResX2R, UNetRes
from isp.model import layers


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
net.load_weights('checkpoints/unet_res_bil_hyp/checkpoint')

# %%

net2 = UNetRes('train', alpha=0.5, weight_norm=False)
t = net2.predict({
    io.dataset_element.MAI_RAW_PATCH: 
      np.zeros([1, 128, 128, 4], dtype=np.float32)
})
net2.load_weights('checkpoints/unet_res_bil_hyp/checkpoint')
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
      tf_record_path_pattern='/home/ron/Downloads/LearnedISP/tfrecord/mai_isp.*.tfrecord'
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

import matplotlib.pyplot as plt

img = np.clip(t['enhanced_rgb'][1], 0, 1) * 255
img = img.astype(np.uint8)
plt.imshow(img)
# %%
