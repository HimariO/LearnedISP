###########################################
# Dataloader for training/validation data #
###########################################

import os
import pdb

import fire
import imageio
import numpy as np
import tensorflow as tf
from PIL import Image
from loguru import logger

from . import dataset
from . mai_tfrecord import (
  extract_bayer_channels,
  load_train_patch,
  load_val_data
)


def get_B5():
  _B5 = tf.keras.applications.EfficientNetB5(
    input_shape=[256, 256, 3], include_top=False)
  
  block3e_add = _B5.layers[188].output  #(32, 32, 64)
  block5g_add = _B5.layers[395].output  #(16, 16, 176)
  block7c_add = _B5.layers[572].output  #(8, 8, 512)
  B5 = tf.keras.Model(_B5.input, [block7c_add, block5g_add, block3e_add])
  return B5


def get_B5_feat(dslr_rgbs: np.ndarray, batch_size=32):
  B5 = get_B5()

  batched = [
    dslr_rgbs[i: i+batch_size] 
    for i in range(0, len(dslr_rgbs), batch_size)
  ]
  feat_list = {
    "block3e_add": [],
    "block5g_add": [],
    "block7c_add": [],
  }
  for i, batch in enumerate(batched):
    if i % 10 == 0:
      print(f"{i}/{len(batched)}")
    feats = B5.predict_on_batch(batch)
    feat_list["block7c_add"].append(feats[0])
    feat_list["block5g_add"].append(feats[1])
    feat_list["block3e_add"].append(feats[2])
  feat_list["block7c_add"] = np.concatenate(feat_list["block7c_add"], axis=0)
  feat_list["block5g_add"] = np.concatenate(feat_list["block5g_add"], axis=0)
  feat_list["block3e_add"] = np.concatenate(feat_list["block3e_add"], axis=0)
  return feat_list


def create_tfrecord(mai_root_dir, tfrecord_dir, type='train'):
  assert type in ['train', 'val', 'eval']
  print(f"mai_root_dir={mai_root_dir}, tfrecord_dir={tfrecord_dir}, type={type}")
  B5 = get_B5()

  def write_tfrecord(output_path, sample_list):
    print(f'Open {output_path}')
    with tf.io.TFRecordWriter(output_path) as writer:
      for sample in sample_list:
        sample_id, raw_img, rgb_img = sample[:3]
        block7c_add, block5g_add, block3e_add = B5.predict(rgb_img[None, ...])
        
        # import pdb; pdb.set_trace()
        # assert short_raw.shape[:2] == long_rgb.shape[:2], \
        #     f"{short_raw.shape} != {long_rgb.shape}"

        example = tf.train.Example(
          features=tf.train.Features(
            feature={
              dataset.MAI_SAMPLE_ID.key:
                dataset.MAI_SAMPLE_ID.to_example_feature_fn(sample_id),
              dataset.MAI_RAW_INPUT.key:
                dataset.MAI_RAW_INPUT.to_example_feature_fn(
                  tf.image.encode_png(raw_img).numpy()),
              dataset.MAI_RAW_INPUT_HEIGHT.key:
                dataset.MAI_RAW_INPUT_HEIGHT.to_example_feature_fn(raw_img.shape[0]),
              dataset.MAI_RAW_INPUT_WIDTH.key:
                dataset.MAI_RAW_INPUT_WIDTH.to_example_feature_fn(raw_img.shape[1]),
              dataset.MAI_RGB_GROUND_TRUTH.key:
                dataset.MAI_RGB_GROUND_TRUTH.to_example_feature_fn(
                  tf.image.encode_png(rgb_img).numpy()),
              
              dataset.MAI_RGB_B5_3E.key:
                dataset.MAI_RGB_B5_3E.to_example_feature_fn(block3e_add.flatten().tolist()),
              dataset.MAI_RGB_B5_3E_SHAPE.key:
                dataset.MAI_RGB_B5_3E_SHAPE.to_example_feature_fn(block3e_add.shape),
              dataset.MAI_RGB_B5_5G.key:
                dataset.MAI_RGB_B5_5G.to_example_feature_fn(block5g_add.flatten().tolist()),
              dataset.MAI_RGB_B5_5G_SHAPE.key:
                dataset.MAI_RGB_B5_5G_SHAPE.to_example_feature_fn(block5g_add.shape),
              dataset.MAI_RGB_B5_7C.key:
                dataset.MAI_RGB_B5_7C.to_example_feature_fn(block7c_add.flatten().tolist()),
              dataset.MAI_RGB_B5_7C_SHAPE.key:
                dataset.MAI_RGB_B5_7C_SHAPE.to_example_feature_fn(block7c_add.shape),
            }))
        writer.write(example.SerializeToString())
  


  try:
    os.makedirs(tfrecord_dir, exist_ok=True)

    samples = []
    tfrecord_counter = 0
    iterator = None
    
    if type == 'train':
      raw_imgs, rgb_imgs = load_train_patch(mai_root_dir, 256, 256, 1)
      # imgs_feat_dict = get_B5_feat(rgb_imgs)
      iterator = zip(
        range(len(raw_imgs)),
        raw_imgs,
        rgb_imgs,
        # imgs_feat_dict['block7c_add'],
        # imgs_feat_dict['block5g_add'],
        # imgs_feat_dict['block3e_add'],
      )
    elif type == 'val' or type == 'eval':
      # iterator = reader.iter_val_or_eval(data_type=type)
      raise NotImplementedError('Ground truth not availble yet')
    else:
      raise RuntimeError(f'Unknown dataset type: {type}')

    for i, raw_rgb_sample in enumerate(iterator):
      samples.append(raw_rgb_sample)
      
      # import pdb; pdb.set_trace()
      if len(samples) >= 2048:
        write_tfrecord(
          os.path.join(tfrecord_dir, f'mai_isp.{tfrecord_counter}.tfrecord'),
          samples
        )
        tfrecord_counter += 1
        samples = []

    if len(samples) > 0:
      write_tfrecord(os.path.join(tfrecord_dir, f'mai_isp.{tfrecord_counter}.tfrecord'),
        samples
      )
  except Exception as e:
    logger.exception(e)


if __name__ == '__main__':
  # train_data, train_answ = load_train_patch(
  #   '/home/ron/Downloads/LearnedISP',
  #   256, 256, 1)
  # import pdb; pdb.set_trace()

  # create_tfrecord('/home/ron/Downloads/LearnedISP', '/home/ron/Downloads/LearnedISP/tfrecord_b5')
  create_tfrecord('/home/ron_zhu/MAI21', '/home/ron_zhu/MAI21/tfrecord_b5')

  # fire.Fire(create_tfrecord)