import os
import json
from typing import List, Dict
from collections import defaultdict

import tensorflow as tf
# import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot
from loguru import logger
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_configs

from isp import metrics, losses, callbacks
from isp import model
from isp.model import base, io, layers
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
    return dict(loss_map)
  
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
    return final_dataset.repeat().prefetch(tf.contrib.data.AUTOTUNE)
  
  def get_val_dataset(self):
    final_dataset = self.get_datasets(self.config.val_datasets)
    return final_dataset
  
  def compilted_model(self, loss_weights=None, model=None):
    model = self.get_model() if model is None else model
    metrics_dict = self.get_metrics()
    losses_dict = self.get_losses()
    logger.info(f"metrics_dict: {metrics_dict}")
    logger.info(f"losses_dict: {losses_dict}")
    
    adam = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config.general['learning_rate'])
    # adam = tfa.optimizers.SWA(adam, start_averaging=8000, average_period=2000)
    model.compile(
      optimizer=adam,
      loss=losses_dict,
      metrics=metrics_dict,
      loss_weights=loss_weights,
    )
    return model
  
  def quantize_model(self, model=None):
    NoOpQuantizeConfig = default_8bit_quantize_configs.NoOpQuantizeConfig

    def apply_quantization(layer):
      no_quan_layers = (
        tf.keras.layers.UpSampling2D,
        tf.keras.layers.Concatenate,
        tf.keras.layers.Lambda)
      if isinstance(layer, no_quan_layers):
        return tfmot.quantization.keras.quantize_annotate_layer(
          layer, quantize_config=NoOpQuantizeConfig())
      # elif isinstance(layer, tf.keras.Model):
      #   layer = tf.keras.models.clone_model(
      #     layer,
      #     clone_function=apply_quantization)
      else:
        return tfmot.quantization.keras.quantize_annotate_layer(layer)

    if model is None:
      model = self.compilted_model()
    annotated_model = tf.keras.models.clone_model(
      model,
      clone_function=apply_quantization)
    custom_objs = {
      "NoOpQuantizeConfig": NoOpQuantizeConfig,
      "ConcatCoordinate": layers.ConcatCoordinate,
    }
    with tf.keras.utils.custom_object_scope(custom_objs):
      quant_aware_model = tfmot.quantization.keras.quantize_model(annotated_model)
    return quant_aware_model


class Experiment:

  def __init__(self, config: ExperimentConfig) -> None:
    self.config = config
    self.builder = ExperimentBuilder(config)
    self.model = None
    self.train_dataset = None
    self.val_dataset = None
  
  def sanity_check(self, model: tf.keras.models.Model, dataset: tf.data.Dataset):
    return
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
      histogram_freq=0,
      write_images=True,
      write_graph=True)
    
    # write_image = callbacks.SaveValImage(log_dir)
    
    ckpt_path = os.path.join(model_dir, 'checkpoint')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
      ckpt_path,
      monitor='val_loss',
      save_best_only=True,
      save_weights_only=True,
    )

    stop_nan = tf.keras.callbacks.TerminateOnNaN()
    return [
      tensorbaord,
      # write_image,
      checkpoint,
      stop_nan
    ]

  def train(self, epoch=None, load_weight=None):
    import os, psutil
    from loguru import logger
    process = psutil.Process(os.getpid())
    logger.info(f"[{self.__class__.__name__}] Train")

    epoch = self.config.general['epoch'] if epoch is None else epoch
    model_dir = self.config.general['model_dir']
    load_weight = self.config.model['pretrain_weight'] if load_weight is None else load_weight
    
    graph = tf.Graph()
    with graph.as_default():
      with tf.Session(graph=graph) as sess:
        tf.keras.backend.set_session(sess)
        self.model = self.builder.compilted_model()
        if load_weight is not None:
          logger.info(f'load_weight: {load_weight}')
          if os.path.exists(os.path.join(load_weight, 'saved_model.pb')):
            # TODO: May be I should add a load_model options?
            self.model = tf.keras.models.load_model(load_weight)
          else:
            self.model.load_weights(load_weight)
        
        self.train_dataset = self.builder.get_train_dataset()
        self.val_dataset = self.builder.get_val_dataset()
        self.sanity_check(self.model, self.val_dataset)
        callback_list = self.callbacks
        
        e_per_loop = 10
        for e in range(0, epoch, e_per_loop):
          self.model.fit(
            self.train_dataset,
            steps_per_epoch=1000,
            epochs=min(epoch, e + e_per_loop),
            initial_epoch=e,
            validation_data=self.val_dataset,
            use_multiprocessing=False,
            workers=1,
            callbacks=callback_list,
          )
          #tf.keras.backend.clear_session()
          logger.debug(f" mem: {process.memory_info().rss / 2**20: .2f}")


class TwoStageExperiment(Experiment):

  def train(self, stage1_epoch=5, skip_stage_1=False, epoch=None, load_weight=None):
    epoch = self.config.general['epoch'] if epoch is None else epoch
    model_dir = self.config.general['model_dir']
    load_weight = self.config.model['pretrain_weight'] if load_weight is None else load_weight
    
    self.train_dataset = self.builder.get_train_dataset()
    self.val_dataset = self.builder.get_val_dataset()

    if load_weight is not None:
      self.sanity_check(self.model, self.val_dataset)
      try:
        self.model.load_weights(load_weight)
      except:
        self.model = tf.keras.models.load_model(load_weight)

    metrics_dict = self.builder.get_metrics()
    losses_dict = self.builder.get_losses()
    adam = tf.optimizers.Adam(learning_rate=self.config.general['learning_rate'])
    callbacks_list = self.callbacks

    """
    Stage 1: Foucs on detail reconstruction, ignore color for now
    """
    if not skip_stage_1:
      self.model.compile(
        optimizer=adam,
        loss=losses_dict,
        metrics=metrics_dict,
        loss_weights={
          io.model_prediction.ENHANCE_RGB: 0.1,
          io.model_prediction.INTER_MID_PRED: 0.9,
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

    """
    Stage 2: Restore RGB image with standar loss func
    """

    self.model.compile(
      optimizer=adam,
      loss=losses_dict,
      metrics=metrics_dict,
      loss_weights={
        io.model_prediction.ENHANCE_RGB: 0.5,
        io.model_prediction.INTER_MID_PRED: 0.5,
      }
    )
    
    for e in range(stage1_epoch, epoch, 10):
      self.model.fit(
        self.train_dataset,
        steps_per_epoch=2000,
        epochs=e + 10,
        initial_epoch=e,
        validation_data=self.val_dataset,
        use_multiprocessing=False,
        workers=1,
        callbacks=callbacks_list,
      )

class ThreeStageExperiment(Experiment):

  def train(self,
            stage1_epoch=5,
            skip_stage_1=False,
            stage2_epoch=5,
            skip_stage_2=False,
            epoch=None,
            load_weight=None):
    epoch = self.config.general['epoch'] if epoch is None else epoch
    model_dir = self.config.general['model_dir']
    load_weight = self.config.model['pretrain_weight'] if load_weight is None else load_weight
    
    self.train_dataset = self.builder.get_train_dataset()
    self.val_dataset = self.builder.get_val_dataset()

    self.sanity_check(self.model, self.val_dataset)
    if load_weight is not None:
      try:
        self.model.load_weights(load_weight)
      except:
        self.model = tf.keras.models.load_model(load_weight)
    
    metrics_dict = self.builder.get_metrics()
    losses_dict = self.builder.get_losses()
    adam = tf.optimizers.Adam(learning_rate=self.config.general['learning_rate'])
    callbacks_list = self.callbacks

    """
    Stage 1: Foucs on detail reconstruction, ignore color for now
    """
    if not skip_stage_1:
      self.model.compile(
        optimizer=adam,
        loss=losses_dict,
        metrics=metrics_dict,
        loss_weights={
          io.model_prediction.ENHANCE_RGB: 1e-6,
          io.model_prediction.INTER_MID_PRED: 1e-4,
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

    """
    Stage 2: Restore RGB image with standar loss func
    """

    stage2_end = stage1_epoch + stage2_epoch
    if not skip_stage_2:
      self.model.compile(
        optimizer=adam,
        loss=losses_dict,
        metrics=metrics_dict,
        loss_weights={
          io.model_prediction.ENHANCE_RGB: 1e-6,
          io.model_prediction.INTER_MID_PRED: 0.8,
          io.model_prediction.INTER_MID_GRAY: 0.2,
        }
      )
      
      for e in range(stage1_epoch, stage2_end, 10):
        self.model.fit(
          self.train_dataset,
          steps_per_epoch=2000,
          epochs=min(stage2_end, e + 10),
          initial_epoch=e,
          validation_data=self.val_dataset,
          use_multiprocessing=False,
          workers=1,
          callbacks=callbacks_list,
        )
    
    """
    Stage 3: Restore RGB image with standar loss func
    """

    self.model.compile(
      optimizer=adam,
      loss=losses_dict,
      metrics=metrics_dict,
      loss_weights={
        io.model_prediction.ENHANCE_RGB: 0.4,
        io.model_prediction.INTER_MID_PRED: 0.4,
        io.model_prediction.INTER_MID_GRAY: 0.2,
      }
    )
    
    for e in range(stage2_end, epoch, 10):
      self.model.fit(
        self.train_dataset,
        steps_per_epoch=2000,
        epochs=min(epoch, e + 10),
        initial_epoch=e,
        validation_data=self.val_dataset,
        use_multiprocessing=False,
        workers=1,
        callbacks=callbacks_list,
      )


class DebugExperiment(Experiment):
  
  def train(self, epoch=None, load_weight=None):
    import os, psutil
    from loguru import logger
    process = psutil.Process(os.getpid())

    epoch = self.config.general['epoch'] if epoch is None else epoch
    model_dir = self.config.general['model_dir']
    load_weight = self.config.model['pretrain_weight'] if load_weight is None else load_weight
    
    self.train_dataset = self.builder.get_train_dataset()
    self.val_dataset = self.builder.get_val_dataset()

    if load_weight is not None:
      logger.info(f'load_weight: {load_weight}')
      self.sanity_check(self.model, self.val_dataset)
      self.model.load_weights(load_weight)
    
    quan_model = self.builder.quantize_model(self.model)
    self.model = self.builder.compilted_model(model=quan_model)
    callback_list = self.callbacks
    
    e_per_loop = 10
    for e in range(0, epoch, 1):
      for i, (x, y) in zip(range(2000), self.train_dataset):
        if i % 50 == 0:
          logger.debug(f"{i}/2000")
        x = {k: v.numpy() for k, v in x.items()}
        y = {k: v.numpy() for k, v in y.items()}
        self.model.train_on_batch(x, y=y, reset_metrics=True)
      self.model.evaluate(self.val_dataset)
      logger.debug(f"epoch[{e}] mem: {process.memory_info().rss / 2**20: .2f}")


class CtxLossExperiment(Experiment):

  def __init__(self, config: ExperimentConfig) -> None:
    loss_weights = {
      io.model_prediction.ENHANCE_RGB: 0.1,
      io.model_prediction.LARGE_FEAT: 0.3,
      io.model_prediction.MID_FEAT: 0.25,
      io.model_prediction.SMALL_FEAT: 0.2,
    }
    self.config = config
    self.builder = ExperimentBuilder(config)
    self.model = self.builder.compilted_model(loss_weights=loss_weights)
    self.train_dataset = None
    self.val_dataset = None
  
  @property
  def callbacks(self):
    model_dir = self.config.general['model_dir']

    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(model_dir, 'logs')
    tensorbaord = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir,
      histogram_freq=1,
      write_images=False,
      write_graph=True)
    
    # NOTE: tf-nightly: tf.summary has no attirbute 'image'
    # write_image = callbacks.SaveValImage(log_dir)
    
    ckpt_path = os.path.join(model_dir, 'checkpoint')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
      ckpt_path,
      monitor='val_loss',
      save_best_only=True,
      save_weights_only=True,
    )

    stop_nan = tf.keras.callbacks.TerminateOnNaN()
    return [
      tensorbaord,
      # write_image,
      checkpoint,
      stop_nan
    ]

  def functionalize_model(self, model):
    if "functional" in self.config.model['type']:
      return model
    
    assert hasattr(model, '_call')
    assert hasattr(model, 'B5')
    shape_in = self.config.model['input_shape']['train']
    model_in = tf.keras.layers.Input(shape=[None, None, shape_in[-1]])
    pred_rgb = model._call(model_in)
    rescale_rgb = tf.clip_by_value(pred_rgb, 0, 1) * 255
    block7c_add, block5g_add, block3e_add = model.B5(rescale_rgb)

    func_input_dict = {
      io.dataset_element.MAI_RAW_PATCH: model_in,
    }
    func_output_dict = {
      io.model_prediction.ENHANCE_RGB: pred_rgb,
      io.model_prediction.LARGE_FEAT: block3e_add,
      io.model_prediction.MID_FEAT: block5g_add,
      io.model_prediction.SMALL_FEAT: block7c_add,
    }
    func_model = tf.keras.Model(func_input_dict, func_output_dict)
    func_model.summary()
    return func_model

  def train(self, epoch=None, load_weight=None, quantize=False):
    import os, psutil
    from loguru import logger
    process = psutil.Process(os.getpid())
    logger.info(f"[{self.__class__.__name__}] Train")

    epoch = self.config.general['epoch'] if epoch is None else epoch
    model_dir = self.config.general['model_dir']
    load_weight = self.config.model['pretrain_weight'] if load_weight is None else load_weight

    if load_weight is not None:
      logger.info(f'load_weight: {load_weight}')
      if os.path.exists(os.path.join(load_weight, 'saved_model.pb')):
        # TODO: May be I should add a load_model options?
        self.model = tf.keras.models.load_model(load_weight)
      else:
        self.model.load_weights(load_weight)
    
    model = (
      self.builder.quantize_model(self.functionalize_model(self.model)) 
      if quantize
      else self.model)
    callback_list = self.callbacks

    self.val_dataset = self.builder.get_val_dataset()
    self.sanity_check(self.model, self.val_dataset)
    self.train_dataset = self.builder.get_train_dataset()
    
    e_per_loop = 10
    for e in range(0, epoch, e_per_loop):
      model.fit(
        self.train_dataset,
        steps_per_epoch=500,
        epochs=min(epoch, e + e_per_loop),
        initial_epoch=e,
        validation_data=self.val_dataset,
        use_multiprocessing=False,
        workers=1,
        callbacks=callback_list,
      )
      logger.debug(f" mem: {process.memory_info().rss / 2**20: .2f}")
