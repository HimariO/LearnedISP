import tensorflow as tf
# import tensorflow.compat.v1 as tf

from tensorflow.python.framework import graph_io
from tensorflow.python.tools import optimize_for_inference_lib


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
    print(frozen_graph_def)
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

if __name__ == "__main__":
  freeze_saved()