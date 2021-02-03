from enum import Enum


class dataset_element:
  SID_RAW = ''
  SID_RGB_GROUND_TRUTH = ''
  MAI_RAW_PATCH = 'mai2021_raw_img_patch'
  MAI_DSLR_PATCH = 'mai2021_dslr_groundtruth'


class model_prediction:
  INTER_MID_GRAY = 'intermidate_grayscale'
  ENHANCE_RGB = 'enhanced_rgb'