from enum import Enum


class dataset_element:
  RGB_IMAGE = 'rgb_image'
  
  SID_RAW = ''
  SID_RGB_GROUND_TRUTH = ''

  MAI_SMAPLE_ID = 'mai2021_sample_id'
  MAI_RAW_PATCH = 'mai2021_raw_img_patch'
  MAI_DSLR_PATCH = 'mai2021_dslr_groundtruth'
  MAI_DSLR_GRAY_PATCH = 'mai2021_dslr_gray_groundtruth'

  MAI_LARGE_FEAT = 'mai2021_dslr_large_feature_map'
  MAI_MID_FEAT = 'mai2021_dslr_mid_feature_map'
  MAI_SMALL_FEAT = 'mai2021_dslr_small_feature_map'


class model_prediction:
  INTER_MID_PRED = 'intermidate_predict'
  INTER_MID_GRAY = 'intermidate_grayscale'
  ENHANCE_RGB = 'enhanced_rgb'
  RGB_CURL = 'rgb_color_curve'

  LARGE_FEAT = 'large_feature_map'
  MID_FEAT = 'mid_feature_map'
  SMALL_FEAT = 'small_feature_map'