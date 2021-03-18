import os
import glob

from imageio.core.util import Image
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import fire
import imageio
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
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


def check_quan_graph():
  with tf.device('/gpu:1'):
    unet_cobi = unet.functional_unet_cobi(mode='quant_train')
    _pred_dict = unet_cobi.predict(np.ones([1, 128, 128, 4], dtype=np.float32))
    for v in _pred_dict:
      print(v.shape)
    
    sess = tf.keras.backend.get_session()
    graph = sess.graph
    sess.run(tf.global_variables_initializer())
    input_nodes, output_nodes = get_io_name(unet_cobi)
    print("input_nodes: ", input_nodes)
    print("output_nodes: ", output_nodes)
    print('-' * 100)

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
      sess,
      graph.as_graph_def(),
      output_nodes[:1]
    )

    with open('cobi_quan.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())


def train_one_stpe():
  with tf.device('/gpu:1'):
    config = experiment.ExperimentConfig('configs/func_unet_05_ctx.json')
    builder = experiment.ExperimentBuilder(config)
    model = builder.compilted_model()
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
    val_iter = val_set.make_one_shot_iterator()

train_one_stpe()