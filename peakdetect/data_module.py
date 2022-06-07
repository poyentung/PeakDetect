import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from peakdetect.utils.dataset_no_simulation import DPdataset_no_simulation
from peakdetect.utils.utils import worker_seed_set, load_classes
from peakdetect.utils.parse_config import parse_data_config


class DPDataModule(pl.LightningDataModule):
    def __init__(self, args, simulation=False):
        super().__init__()
        self.args = args
        self.data_config = parse_data_config(self.args['data'])
        self.struct_path = self.data_config["structure"]
        self.class_names = load_classes(self.data_config["class_names"])
        self.all_euler_angles_path = self.data_config["all_euler_angles_path"]
        self.simulation = simulation

    def prepare_data(self):
        # make dataframe for euler_angles
        angle_list = list()
        with open(self.all_euler_angles_path) as f:
            for angle in f.readlines()[2:]:
                angle.replace('\n','')
                angles = angle.split()
                angle_list.append(angles)
        self.euler_angles = pd.DataFrame(angle_list, columns=['z1','x','z2']).astype(float)

        if not self.simulation:
            self.dp_images_path_train = self.data_config["dp_images_path_train"]
            self.targets_path_train = self.data_config["targets_path_train"]
            self.euler_angles_train = self.data_config['euler_angles_path_train']
            self.dp_images_path_valid = self.data_config["dp_images_path_valid"]
            self.targets_path_valid = self.data_config["targets_path_valid"]
            self.euler_angles_valid = self.data_config['euler_angles_path_valid']

            # self.dp_images = np.load(self.dp_images_path)
            # self.targets = pd.read_csv(self.targets_path)

            # total_num = self.dp_images.shape[0]
            # self.idx_train, self.idx_val = train_test_split(range(total_num),test_size=self.args['test_data_ratio'])
            # self.idx_train = sorted(self.idx_train)
            # self.idx_val = sorted(self.idx_val)

            # # As slicing the dataframe, we have to reset the id in the target_dataframe
            # self.train_targets = self.targets.loc[self.targets['id'].isin(self.idx_train)].reset_index(drop=True)
            # self.valid_targets = self.targets.loc[self.targets['id'].isin(self.idx_val)].reset_index(drop=True)

            # # disable the warning for SettingWithCopyWarning
            # pd.options.mode.chained_assignment = None 
            # for target in (self.train_targets, self.valid_targets):
            #     unique_ids = target['id'].unique()
            #     for i, idx in enumerate(unique_ids):
            #         target.loc[target['id']==idx,'id'] = i
        
    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            self.train = DPdataset_no_simulation(struct_filename=self.struct_path, 
                                                euler_angle_filename=self.euler_angles_train,
                                                dp_image_path=self.dp_images_path_train,
                                                targets_path=self.targets_path_train,
                                                pattern_size = self.data_config['pattern_size'],
                                                pattern_sigma = self.data_config['pattern_sigma'],
                                                reciprocal_radius = self.data_config['reciprocal_radius'],
                                                acceleration_voltage=self.data_config['acceleration_voltage'],
                                                max_excitation_error=self.data_config['max_excitation_error'])

            self.valid = DPdataset_no_simulation(struct_filename=self.struct_path, 
                                                euler_angle_filename=self.euler_angles_valid,
                                                dp_image_path=self.dp_images_path_valid,
                                                targets_path=self.targets_path_valid,
                                                pattern_size = self.data_config['pattern_size'],
                                                pattern_sigma = self.data_config['pattern_sigma'],
                                                reciprocal_radius = self.data_config['reciprocal_radius'],
                                                acceleration_voltage=self.data_config['acceleration_voltage'],
                                                max_excitation_error=self.data_config['max_excitation_error'])

        if stage == "test" or stage is None:
            self.test = DPdataset_no_simulation(struct_filename=self.struct_path, 
                                                euler_angle_filename=self.euler_angles_valid,
                                                dp_image_path=self.dp_images_path_valid,
                                                targets_path=self.targets_path_valid,
                                                pattern_size = self.data_config['pattern_size'],
                                                pattern_sigma = self.data_config['pattern_sigma'],
                                                reciprocal_radius = self.data_config['reciprocal_radius'],
                                                acceleration_voltage=self.data_config['acceleration_voltage'],
                                                max_excitation_error=self.data_config['max_excitation_error'])

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.args['batch_size'],
                          shuffle=True,
                          num_workers=self.args['n_cpu'],
                          pin_memory=True,
                          collate_fn=self.train.collate_fn,
                          worker_init_fn=worker_seed_set)
    
    def val_dataloader(self):
        return DataLoader(self.valid,
                          batch_size=self.args['batch_size'],
                          shuffle=False,
                          num_workers=self.args['n_cpu'],
                          pin_memory=True,
                          collate_fn=self.train.collate_fn,
                          worker_init_fn=worker_seed_set)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.args['batch_size'],
                          shuffle=False,
                          num_workers=self.args['n_cpu'],
                          pin_memory=True,
                          collate_fn=self.train.collate_fn,
                          worker_init_fn=worker_seed_set)
    def predict_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.args['batch_size'],
                          shuffle=False,
                          num_workers=self.args['n_cpu'],
                          pin_memory=True,
                          collate_fn=self.train.collate_fn,
                          worker_init_fn=worker_seed_set)
