import os
import json
from typing import List, Dict
from collections import defaultdict

import tensorflow as tf
from loguru import logger
from isp import metrics, losses, callbacks
from isp import model
from isp.model import base, io
from isp.data import dataset, preprocess


class ExperimentConfig:

  def __init__(self, config_or_json):
    if isinstance(config_or_json, dict):
      self.raw_config = config_or_json
    else:
      with open(config_or_json, mode='r') as f:
        self.raw_config = json.load(f)
    self.set_attribute(self.raw_config)
  
  def set_attribute(self, cfg_json):
    template = {
      'general': {
        'learning_rate': 0.0001,
        'model_dir': '',
        'log_dir': '',
        'epoch': 1000,
        'scheduler': 'cycle',
        'batch_size': 8,
      },
      'model': {
        'type': lambda x: x in base._TAG_TO_PREDICTION_MODEL,
        'init_param': {
          'args': [],
          'kwargs': {},
        },
        'pretrain_weight': lambda x: x is None or os.path.exists(x),
        'input_shape': {
          'train': [1, 2, 3, 4],
          'inference': lambda x: x is None or isinstance(x, list),
        }
      },
      'losses': [
        {
          'type': lambda x: x in losses._TAG_TO_PREDICTION_LOSS,
          'args': [],
          'kwargs': {},
          'target_output': lambda x: x is None or x in io.model_prediction
        }
      ],
      'metrics': [
        {
          'type': lambda x: x in metrics._TAG_TO_PREDICTION_METRIC,
          'args': [],
          'kwargs': {},
          'target_output': lambda x: x is None or x in io.model_prediction
        }
      ],
      'train_datasets': [
        {
          'type': lambda x: x in dataset._TAG_TO_DATASET,
          'args': [],
          'kwargs': {},
        }
      ],
      'val_datasets': [
        {
          'type': lambda x: x in dataset._TAG_TO_DATASET,
          'args': [],
          'kwargs': {},
        }
      ],
      'preprocessing': [
        {
          'type': lambda x: x in preprocess._TAG_TO_PREPROCESSING_CALLABLE,
          'args': [],
          'kwargs': {}
        }
      ],
      'callback':[
        {
          'type': lambda x: x in [],
          'args': [],
          'kwargs': {}
        }
      ]
    }
    ExperimentConfig.check_dict_struct(template, cfg_json)
    self.general = cfg_json['general']
 
    self.model = cfg_json['model']
    self.losses = cfg_json['losses']
    self.metrics = cfg_json['metrics']
 
    self.preprocessing = cfg_json['preprocessing']
    self.train_datasets = cfg_json['train_datasets']
    self.val_datasets = cfg_json['val_datasets']
    
  @staticmethod
  def check_dict_struct(a, b):
    """
    Check the structure and content type of dictionary "a" and "b" is the same.
    'a' is reference template, 'b' is new sample. If isn't, raise AssertionError.
    """
    if callable(a):
      assert a(b), f'{a.__name__}({b}) failed'
    elif a is not None:
      assert type(a) == type(b), \
        f"type of config doesn't match template: {type(a)} vs {type(b)}"

    if isinstance(a, dict):
      for k, v in a.items():
        assert k in b, f"keyword \"{k}\" not in {b}!"
        if callable(v):
          v.__name__ = f'[{k}] {v.__name__}'
      ExperimentConfig.check_dict_struct(v, b[k])
    elif isinstance(a, list):
      for v, vb in zip(a, b):
        if callable(v):
          v.__name__ = f'[{k}] {v.__name__}'
        ExperimentConfig.check_dict_struct(v, vb)


class ExperimentBuilder:

  def __init__(self, config: ExperimentConfig):
    self.config = config
  
  def get_model(self) -> tf.keras.Model:
    # cfg = self.config['model']
    cfg = self.config.model
    model_class = base.get_model(cfg['type'])
    model = model_class(
      *cfg['init_param']['args'],
      **cfg['init_param']['kwargs']
    )
    return model

  def get_losses(self) -> Dict[str, losses.PredictionLossBase]:
    cfg = self.config.losses
    loss_map = {}
    for loss_cfg in cfg:
      loss_class = losses.get_prediction_loss(loss_cfg['type'])
      loss_func = loss_class(*loss_cfg['args'], **loss_cfg['kwargs'])
      loss_map[loss_cfg['target_output']] = loss_func
    return loss_map
  
  def get_metrics(self) -> Dict[str, metrics.PredictionMetricBase]:
    cfg = self.config.metrics
    metric_map = defaultdict(list)
    for metric_cfg in cfg:
      metric_class = metrics.get_prediction_metric(metric_cfg['type'])
      metric_func = metric_class(*metric_cfg['args'], **metric_cfg['kwargs'])
      metric_map[metric_cfg['target_output']].append(metric_func)
    return dict(metric_map)
  
  def get_preprocess(self) -> List[preprocess.DataPreprocessingBase]:
    cfg = self.config.preprocessing
    transforms = []
    for preprocessor in cfg:
      process_class = preprocess.get_data_preprocessing_callable(preprocessor['type'])
      transform = process_class(
        *preprocessor['args'], **preprocessor['kwargs'])
      transforms.append(transform)
    return transforms
  
  def get_datasets(self, cfg, preprocess=False) -> tf.data.Dataset:
    transforms = self.get_preprocess() if preprocess else []
    datasets = []
    for dataset_cfg in cfg:
      dataset_cls = dataset.get_dataset(dataset_cfg['type'])
      dataset_obj = dataset_cls(
        *dataset_cfg['args'],
        **dataset_cfg['kwargs'],
        data_preprocessing_callables=transforms)
      
      tf_dataset = dataset_obj.create_dataset(
        batch_size=self.config.general['batch_size'],
        num_readers=4,
        num_parallel_calls=8
      )
      datasets.append(tf_dataset)
    return tf.data.experimental.sample_from_datasets(datasets)
  
  def get_train_dataset(self):
    final_dataset = self.get_datasets(self.config.train_datasets, preprocess=True)
    return final_dataset.repeat().prefetch(tf.data.AUTOTUNE)
  
  def get_val_dataset(self):
    final_dataset = self.get_datasets(self.config.val_datasets)
    return final_dataset
  
  def compilted_model(self):
    model = self.get_model()
    metrics_dict = self.get_metrics()
    losses_dict = self.get_losses()
    logger.info(f"metrics_dict: {metrics_dict}")
    logger.info(f"losses_dict: {losses_dict}")
    
    adam = tf.optimizers.Adam(learning_rate=self.config.general['learning_rate'])
    model.compile(
      optimizer=adam,
      loss=losses_dict,
      metrics=metrics_dict,
    )
    return model


class Experiment:

  def __init__(self, config: ExperimentConfig) -> None:
    self.config = config
    self.builder = ExperimentBuilder(config)
    self.model = self.builder.compilted_model()
    self.train_dataset = None
    self.val_dataset = None
  
  def sanity_check(self, model: tf.keras.models.Model, dataset: tf.data.Dataset):
    for x, y in dataset:
      pred = model.predict(x)
      np_label = y[io.model_prediction.ENHANCE_RGB].numpy()
      print(np_label.max(), np_label.min())
      
      np_pred = pred[io.model_prediction.ENHANCE_RGB]
      print(np_pred.max(), np_pred.min())
      # import pdb; pdb.set_trace()
      break
    model.summary()
  
  @property
  def callbacks(self):
    model_dir = self.config.general['model_dir']

    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(model_dir, 'logs')
    tensorbaord = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir,
      write_images=True,
      write_graph=True)
    
    write_image = callbacks.SaveValImage(log_dir)
    
    ckpt_path = os.path.join(model_dir, 'checkpoint')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
      ckpt_path,
      monitor='val_loss',
      save_best_only=False,
    )
    return [
      tensorbaord,
      write_image,
      checkpoint,
    ]

  def train(self, epoch=None, load_weight=None):
    epoch = self.config.general['epoch'] if epoch is None else epoch
    model_dir = self.config.general['model_dir']
    load_weight = self.config.model['pretrain_weight'] if load_weight is None else load_weight
    
    self.train_dataset = self.builder.get_train_dataset()
    self.val_dataset = self.builder.get_val_dataset()

    if load_weight is not None:
      self.sanity_check(self.model, self.val_dataset)
      self.model.load_weights(load_weight)
    
    self.model.fit(
      self.train_dataset,
      steps_per_epoch=2000,
      epochs=epoch,
      validation_data=self.val_dataset,
      use_multiprocessing=False,
      workers=1,
      callbacks=self.callbacks,
    )


class TwoStageExperiment(Experiment):

  def train(self, stage1_epoch=5, epoch=None, load_weight=None):
    epoch = self.config.general['epoch'] if epoch is None else epoch
    model_dir = self.config.general['model_dir']
    load_weight = self.config.model['pretrain_weight'] if load_weight is None else load_weight
    
    self.train_dataset = self.builder.get_train_dataset()
    self.val_dataset = self.builder.get_val_dataset()

    if load_weight is not None:
      self.sanity_check(self.model, self.val_dataset)
      self.model.load_weights(load_weight)

    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(model_dir, 'logs')
    tensorbaord = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir,
      write_images=True,
      write_graph=True)
    
    write_image = callbacks.SaveValImage(log_dir)
    
    ckpt_path = os.path.join(model_dir, 'checkpoint')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
      ckpt_path,
      monitor='val_loss',
      save_best_only=False,
    )

    
    metrics_dict = self.builder.get_metrics()
    losses_dict = self.builder.get_losses()
    adam = tf.optimizers.Adam(learning_rate=self.config.general['learning_rate'])
    callbacks_list = self.callbacks

    self.model.compile(
      optimizer=adam,
      loss=losses_dict,
      metrics=metrics_dict,
      loss_weights={
        io.model_prediction.ENHANCE_RGB: 0,
        io.model_prediction.INTER_MID_GRAY: 1,
      }
    )
    
    self.model.fit(
      self.train_dataset,
      steps_per_epoch=2000,
      epochs=stage1_epoch,
      validation_data=self.val_dataset,
      use_multiprocessing=False,
      workers=1,
      callbacks=callbacks_list,
    )

    self.model.compile(
      optimizer=adam,
      loss=losses_dict,
      metrics=metrics_dict,
      loss_weights={
        io.model_prediction.ENHANCE_RGB: 0.8,
        io.model_prediction.INTER_MID_GRAY: 0.2,
      }
    )
    
    self.model.fit(
      self.train_dataset,
      steps_per_epoch=2000,
      initial_epoch=stage1_epoch,
      epochs=epoch,
      validation_data=self.val_dataset,
      use_multiprocessing=False,
      workers=1,
      callbacks=callbacks_list,
    )