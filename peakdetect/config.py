from dataclasses import dataclass

@dataclass
class DataFiles:
  structure: str
  dp_images_path_train:str
  targets_path_train: str
  euler_angles_path_train: str
  dp_images_path_valid: str
  targets_path_valid: str
  euler_angles_path_valid: str
  class_names: str

@dataclass
class SimulationParams:
  pattern_size: int
  pattern_sigma: float
  reciprocal_radius: float
  acceleration_voltage: float
  max_excitation_error: float
  inimum_intensity: float

@dataclass
class Params:
  model: str
  model_name: str
  version: str
  test_data_ratio: float
  batch_size: int
  precision_for_training: int
  num_epoch: int
  verbose: str
  n_cpu: int
  iou_thres: float
  conf_thres: float
  nms_thres: float
  seed: int

@dataclass
class TrainerParams:
  max_epochs: int
  accelerator: str
  devices: int

@dataclass
class Checkpoints:
  checkpoints: str
  monitor: str
  every_n_epochs: int
  mode: str
  filename: str

@dataclass
class Path:
  logs: str

@dataclass
class PeakConfig:
    data_files:DataFiles
    simulation: SimulationParams
    params:Params
    trainer:TrainerParams
    checkpoints: Checkpoints
    path:Path
