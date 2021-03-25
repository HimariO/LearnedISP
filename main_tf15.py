import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf

from loguru import logger
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import optimize_for_inference_lib

from isp.model.unet import (
  UNet,
  UNetGrid,
  UNetResX2R,
  UNetRes,
  functional_unet,
  functional_unet_grid)
from isp.model import io


def quantize_saved_model(saved_model_dir, input_nodes, output_nodes):
  graph_def = tf.Graph()
  with tf.Session(graph=graph_def) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
    tf.contrib.quantize.create_training_graph(input_graph=sess.graph)

    input_tensor = [
      sess.graph.get_tensor_by_name(out_name)
      for out_name in input_nodes]
    output_tensor = [
      sess.graph.get_tensor_by_name(out_name)
      for out_name in output_nodes]
    print(input_tensor)
    print(output_tensor)
    grads = tf.gradients(output_tensor[0], input_tensor[0])

    grad_val = sess.run(grads, feed_dict={
      input_tensor[0]: np.zeros([1, 544, 960, 4], dtype=np.float32)
    })
    print(grad_val)


def freeze(saved_model_dir, input_nodes, output_nodes, save_file):
  graph_def = tf.Graph()
  with tf.Session(graph=graph_def) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_nodes
    )
    frozen_graph_def = optimize_for_inference_lib.optimize_for_inference(
        frozen_graph_def,
        input_nodes,
        output_nodes,
        tf.float32.as_datatype_enum
    )
    # print(frozen_graph_def)
    with open(save_file, 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())

def freeze_saved():
  input_nodes = ['serving_default_input_1']
  output_nodes = ['StatefulPartitionedCall']
  saved_model_dir = 'checkpoints/unet_05_grid/saved'
  save_file = 'tf15.pb'
  freeze(saved_model_dir, input_nodes, output_nodes, save_file)


def saved_to_tflite():
  path = 'checkpoints/unet_05_grid/saved'
  converter = tf.lite.TFLiteConverter.from_saved_model(path)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.allow_custom_ops = True
  tflite_model = converter.convert()
  print('[END]')

  with open('tf15.tflite', mode='wb') as f:
      f.write(tflite_model)


def pb_to_tflite():
  path = 'frozen_graph.pb'
  converter = tf.lite.TFLiteConverter.from_frozen_graph(
    path, input_arrays=["x"], output_arrays=["model/quant_lambda/DepthToSpace"])
  # converter = tf.lite.TFLiteConverter.from_frozen_graph(
  #   path, input_arrays=["x"], output_arrays=["model/lambda/DepthToSpace"])
  
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  # converter.allow_custom_ops = True
  # converter.inference_input_type = tf.uint8
  # converter.quantized_input_stats = {"x": (0, 255)}
  # converter.inference_type = tf.uint8
  # converter.target_spec = tf.lite.TargetSpec(supported_ops=tf.lite.OpsSet.TFLITE_BUILTINS_INT8)
  tflite_model = converter.convert()
  print('[END]')

  with open('tf15.tflite', mode='wb') as f:
      f.write(tflite_model)


def load_keras_h5(in_size=[256, 256]):
  # tf.compat.v1.disable_eager_execution()
  tf.enable_control_flow_v2()
  
  def get_keras_model():
    # net = UNet('export', alpha=1.0)
    # payload = np.random.normal(size=[1, *in_size, 4]).astype(np.float32)
    # net.predict({
    #   io.dataset_element.MAI_RAW_PATCH: payload
    # })
    # # net.load_weights(model_dir)
    
    # tf_pred = net.predict({
    #   io.dataset_element.MAI_RAW_PATCH: payload
    # })
    # # remove_weight_norm(net)
    
    # x = tf.keras.Input(shape=[*in_size, 4], batch_size=1, dtype=tf.float32)
    # y = net._call(x)
    # y = tf.cast(tf.clip_by_value(y, 0.0, 1.0) * 255, tf.uint8)
    functional_net = functional_unet(input_shape=[*in_size, 4], mode='functional')
    functional_net.predict(np.zeros([1, *in_size, 4]))
    return functional_net
  
  sess = tf.Session()
  with sess.as_default():
    with sess.graph.as_default():
      tf.keras.backend.set_session(sess)
      # tf.keras.models.load_model('test_h5.h5')
      model = get_keras_model()
      # model.load_weights('test_h5.h5')

      payload = np.ones([1, *in_size, 4]).astype(np.float32)
      pred = model.predict({
        io.dataset_element.MAI_RAW_PATCH: payload,
        # 'input_1': payload,
      })
      print(pred.mean(), pred.shape)
      graph_ops = [n for n in tf.get_default_graph().as_graph_def().node]
      # for op in graph_ops:
      #   print(op.name)
      print('-' * 100)

      graph = sess.graph
      tf.contrib.quantize.create_training_graph(input_graph=graph)
      # tf.contrib.quantize.create_eval_graph(input_graph=graph)
      sess.run(tf.global_variables_initializer())
      
      if isinstance(model.input, dict):
        input_nodes = [ip.name.split(':')[0] for ip in model.input.values()]
      else:
        input_nodes = [model.input.name.split(':')[0]]
      if isinstance(model.output, dict):
        output_nodes = [op.name.split(':')[0] for op in model.output.values()]
      else:
        output_nodes = [model.output.name.split(':')[0]]
      
      # frozen_graph_def = graph.as_graph_def()
      frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        graph.as_graph_def(),
        output_nodes
        # ['lambda_1/DepthToSpace'],
      )
      # frozen_graph_def = optimize_for_inference_lib.optimize_for_inference(
      #   frozen_graph_def,
      #   input_nodes,
      #   output_nodes,
      #   tf.float32.as_datatype_enum
      # )

      # for node in frozen_graph_def.node:
      #   if node.op == 'RefSwitch':
      #     node.op = 'Switch'
      #     for index in range(len(node.input)):
      #       if 'moving_' in node.input[index]:
      #         node.input[index] = node.input[index] + '/read'
      #   elif node.op == 'AssignSub':
      #     node.op = 'Sub'
      #     if 'use_locking' in node.attr: del node.attr['use_locking']

      with open('h5_quan.pb', 'wb') as f:
        # import pdb; pdb.set_trace()
        # f.write(graph.as_graph_def().SerializeToString())
        f.write(frozen_graph_def.SerializeToString())
      
      print("input_nodes: ", input_nodes)
      print("output_nodes: ", output_nodes)
      print('-' * 100)
      # converter = tf.lite.TFLiteConverter.from_session(
      #   sess, list(model.input.values()), list(model.output.values()))
      converter = tf.lite.TFLiteConverter.from_frozen_graph(
        'h5_quan.pb', input_arrays=input_nodes, output_arrays=output_nodes)
      converter.inference_input_type = tf.uint8
      converter.inference_type = tf.uint8
      converter.quantized_input_stats = {io.dataset_element.MAI_RAW_PATCH: (0, 255)}
      tflite_model = converter.convert()
      open("converted_model.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
  # freeze_saved()
  # pb_to_tflite()
  # quantize_saved_model(
  #   'checkpoints/unet_05_grid/saved',
  #   ['serving_default_input_1:0'],
  #   ['StatefulPartitionedCall:0'],
  # )
  with logger.catch(reraise=True):
    load_keras_h5()