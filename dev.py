# %%

import os
import glob

from imageio.core.util import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import fire
import imageio
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
from PIL import Image
from loguru import logger

from isp import metrics
from isp import losses
from isp import callbacks
from isp import experiment
from isp.model import io
from isp.model import unet
from isp.data import dataset


def get_io_name(model):
  if isinstance(model.input, dict):
    input_nodes = [ip.name.split(':')[0] for ip in model.input.values()]
  else:
    input_nodes = [model.input.name.split(':')[0]]
  if isinstance(model.output, dict):
    output_nodes = [op.name.split(':')[0] for op in model.output.values()]
  else:
    output_nodes = [model.output.name.split(':')[0]]
  return input_nodes, output_nodes

# %%

config = experiment.ExperimentConfig('configs/func_unet_05_ctx.json')
builder = experiment.ExperimentBuilder(config)

loss_weights = {
  io.model_prediction.ENHANCE_RGB: 0.01,
  io.model_prediction.LARGE_FEAT: 0.3,
  io.model_prediction.MID_FEAT: 0.25,
  io.model_prediction.SMALL_FEAT: 0.2,
}
model = builder.compilted_model(loss_weights=loss_weights)
x = {
  io.dataset_element.MAI_RAW_PATCH: np.ones([1, 128, 128, 4], dtype=np.float32),
}
y = {
  io.model_prediction.ENHANCE_RGB: np.ones([1, 256, 256, 3], dtype=np.float32),
  io.model_prediction.LARGE_FEAT: np.ones([1, 32, 32, 64], dtype=np.float32),
  io.model_prediction.MID_FEAT: np.ones([1, 16, 16, 176], dtype=np.float32),
  io.model_prediction.SMALL_FEAT: np.ones([1, 8, 8, 512], dtype=np.float32),
}
out = model.train_on_batch(x, y)
print(out)
print(model.metrics_names)

val_set = builder.get_val_dataset()

# %%

model.load_weights('checkpoints/func_unet_05_grid/checkpoint')

# %%
d = next(iter(val_set))
# for i, d in enumerate(val_set):
for i in range(500):
  loss_metrics = model.train_on_batch(d[0], d[1])
  if i % 10 == 0:
    print(f'[{i}]')
    for n, v in zip(model.metrics_names, loss_metrics):
      if n == 'enhanced_rgb_cache_output':
        continue
      print(n, v)

# %%
