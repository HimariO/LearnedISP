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


def extract_bayer_channels(raw):

  # Reshape the input bayer image
  ch_B  = raw[1::2, 1::2]
  ch_Gb = raw[0::2, 1::2]
  ch_R  = raw[0::2, 0::2]
  ch_Gr = raw[1::2, 0::2]

  RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
  RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

  return RAW_combined


def load_val_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE):

  val_directory_dslr = dataset_dir + 'val/' + dslr_dir
  val_directory_phone = dataset_dir + 'val/' + phone_dir

  # get the image format (e.g. 'png')
  format_dslr = str.split(os.listdir(val_directory_dslr)[0],'.')[-1]

  # determine validation image numbers by listing all files in the folder
  NUM_VAL_IMAGES = len([name for name in os.listdir(val_directory_phone)
               if os.path.isfile(os.path.join(val_directory_phone, name))])

  val_data = np.zeros((NUM_VAL_IMAGES, PATCH_WIDTH, PATCH_HEIGHT, 4))
  val_answ = np.zeros((NUM_VAL_IMAGES, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3))

  for i in range(0, NUM_VAL_IMAGES):

    I = np.asarray(imageio.imread((val_directory_phone + str(i) + '.png')))
    I = extract_bayer_channels(I)
    val_data[i, :] = I

    I = Image.open(val_directory_dslr + str(i) + '.' + format_dslr)
    I = np.array(I.resize((int(I.size[0] * DSLR_SCALE / 2), int(I.size[1] * DSLR_SCALE / 2)), resample=Image.BICUBIC))
    I = np.float16(np.reshape(I, [1, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3])) / 255
    val_answ[i, :] = I

  return val_data, val_answ


def load_train_patch(dataset_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE):

  train_directory_dslr = os.path.join(dataset_dir, 'MAI2021_LearnedISP_train', 'fujifilm')
  train_directory_phone = os.path.join(dataset_dir, 'MAI2021_LearnedISP_train', 'mediatek_raw')

  # get the image format (e.g. 'png')
  format_dslr = str.split(os.listdir(train_directory_dslr)[0],'.')[-1]

  # determine training image numbers by listing all files in the folder
  NUM_TRAINING_IMAGES = len([name for name in os.listdir(train_directory_phone)
                 if os.path.isfile(os.path.join(train_directory_phone, name))])

  TRAIN_IMAGES = np.arange(0, NUM_TRAINING_IMAGES)

  train_data = []
  train_answ = []
  # train_data = np.zeros((PATCH_WIDTH, PATCH_HEIGHT, 4))
  # train_answ = np.zeros((int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3))

  for i, img in enumerate(TRAIN_IMAGES):
    I = np.asarray(imageio.imread(os.path.join(train_directory_phone, str(img) + '.png')))
    I = extract_bayer_channels(I)
    # train_data[i, :] = I
    train_data.append(I)
    # import pdb; pdb.set_trace()

    I = Image.open(os.path.join(train_directory_dslr, str(img) + '.' + format_dslr))
    # I = np.array(I.resize((int(I.size[0] * DSLR_SCALE / 2), int(I.size[1] * DSLR_SCALE / 2)), resample=Image.BICUBIC))
    I = np.asarray(I)
    I = I.reshape([
      int(PATCH_WIDTH * DSLR_SCALE),
      int(PATCH_HEIGHT * DSLR_SCALE),
      3
    ])
    # I = np.float32(I) / 255
    # train_answ[i, :] = I
    train_answ.append(I)
  
  train_data = np.stack(train_data, axis=0)
  train_answ = np.stack(train_answ, axis=0)
  return train_data, train_answ


def create_tfrecord(mai_root_dir, tfrecord_dir, type='train'):
  assert type in ['train', 'val', 'eval']
  print(f"mai_root_dir={mai_root_dir}, tfrecord_dir={tfrecord_dir}, type={type}")

  def write_tfrecord(output_path, sample_list):
    print(f'Open {output_path}')
    with tf.io.TFRecordWriter(output_path) as writer:
      for sample in sample_list:
        raw_img, rgb_img = sample
        
        # assert short_raw.shape[:2] == long_rgb.shape[:2], \
        #     f"{short_raw.shape} != {long_rgb.shape}"

        example = tf.train.Example(
          features=tf.train.Features(
            feature={
              dataset.MAI_RAW_INPUT.key:
                dataset.MAI_RAW_INPUT.to_example_feature_fn(
                  tf.image.encode_png(raw_img).numpy()),
              dataset.MAI_RAW_INPUT_HEIGHT.key:
                dataset.MAI_RAW_INPUT_HEIGHT.to_example_feature_fn(rgb_img.shape[0]),
              dataset.MAI_RAW_INPUT_WIDTH.key:
                dataset.MAI_RAW_INPUT_WIDTH.to_example_feature_fn(rgb_img.shape[1]),
              dataset.MAI_RGB_GROUND_TRUTH.key:
                dataset.MAI_RGB_GROUND_TRUTH.to_example_feature_fn(
                  tf.image.encode_png(rgb_img).numpy()),
            }))
        writer.write(example.SerializeToString())

  try:
    os.makedirs(tfrecord_dir, exist_ok=True)

    samples = []
    tfrecord_counter = 0
    iterator = None
    
    if type == 'train':
      raw_imgs, rgb_imgs = load_train_patch(mai_root_dir, 256, 256, 1)
      iterator = zip(raw_imgs, rgb_imgs)
    elif type == 'val' or type == 'eval':
      # iterator = reader.iter_val_or_eval(data_type=type)
      raise NotImplementedError('Ground truth not availble yet')
    else:
      raise RuntimeError(f'Unknown dataset type: {type}')

    for i, raw_rgb_sample in enumerate(iterator):
      samples.append(raw_rgb_sample)
      
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

  create_tfrecord('/home/ron/Downloads/LearnedISP', '/home/ron/Downloads/LearnedISP/tfrecord')

  # fire.Fire(create_tfrecord)