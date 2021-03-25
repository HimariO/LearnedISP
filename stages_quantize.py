# %%

import os
import glob

from imageio.core.util import Image
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

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
from isp.model import io, layers
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
    next_element = val_iter.get_next()
    session = tf.keras.backend.get_session()
    d = session.run(next_element)
    print(d)

# %%
with tf.device('/gpu:0'):
  graph = tf.Graph()
  with graph.as_default():
    with tf.Session(graph=graph) as sess:
      tf.keras.backend.set_session(sess)
      
      config = experiment.ExperimentConfig('configs/func_unet_05_ctx.json')
      builder = experiment.ExperimentBuilder(config)

      loss_weights = {
        io.model_prediction.ENHANCE_RGB: 0.1,
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
      val_iter = val_set.make_one_shot_iterator()
      next_element = val_iter.get_next()
      session = tf.keras.backend.get_session()

      # %%

      model.load_weights('checkpoints/func_unet_05_grid_0320/checkpoint')
      # model.evaluate(val_set, steps=2000//32)

      # %%
      checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/stages_quan',
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=True,
      )
      train_set = builder.get_val_dataset()
      model.fit(
        train_set,
        steps_per_epoch=500,
        epochs=1000,
        initial_epoch=0,
        validation_data=val_set,
        validation_steps=60,
        use_multiprocessing=False,
        workers=1,
        callbacks=[checkpoint],
      )
      # %%
      for i in range(100):
        d = session.run(next_element)
        loss_metrics = model.train_on_batch(d[0], d[1])
        if i % 10 == 0:
          print(f'[{i}]')
          for n, v in zip(model.metrics_names, loss_metrics):
            print(n, v)

# %%

# model_2 = builder.compilted_model(loss_weights=loss_weights)

# # %%

# b5 = model.layers[-4]
# b5_2 = model_2.layers[-4]

# for i, (l1, l2) in enumerate(zip(b5.layers, b5_2.layers)):
#   print(f"[{i}]", l1)
#   for w1, w2 in zip(l1.get_weights(), l2.get_weights()):
#     print(np.abs(w1 - w2).mean())
#   if i > 20:
#     break
# %%
# train_one_stpe()