from enum import Enum


class dataset_element:
  RGB_IMAGE = 'rgb_image'
  
  SID_RAW = ''
  SID_RGB_GROUND_TRUTH = ''

  MAI_SMAPLE_ID = 'mai2021_sample_id'
  MAI_RAW_PATCH = 'mai2021_raw_img_patch'
  MAI_DSLR_PATCH = 'mai2021_dslr_groundtruth'
  MAI_DSLR_GRAY_PATCH = 'mai2021_dslr_gray_groundtruth'


class model_prediction:
  INTER_MID_PRED = 'intermidate_grayscale'
  ENHANCE_RGB = 'enhanced_rgb'
