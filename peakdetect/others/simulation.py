import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyxem as pxm
import diffpy.structure
import os


from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.utils.shape_factor_models import sinc

class DiffGen(object):
    def __init__(self, 
                 struct_file, 
                 angle_file, 
                 pattern_size=100,
                 reciprocal_radius=2,
                 acceleration_voltage=200,
                 max_excitation_error=0.03,
                 spot_sigma=1.5):
        
        self.struct_file = struct_file
        self.struct = diffpy.structure.loadStructure(struct_file)
        self.angle_file = angle_file
        self.pattern_size = pattern_size
        self.half_pattern_size = self.pattern_size // 2
        self.reciprocal_radius = reciprocal_radius
        self.calibration = self.reciprocal_radius / self.half_pattern_size
        self.acceleration_voltage = acceleration_voltage
        self.max_excitation_error = max_excitation_error
        self.sigma = spot_sigma

        # Create the dataframe for euler angles
        angle_list = list()
        with open(self.angle_file) as f:
            for angle in f.readlines()[2:]:
                angle.replace('\n','')
                angles = angle.split()
                angle_list.append(angles)
        self.angle_df = pd.DataFrame(angle_list, columns=['z1','x','z2']).astype(float)
    
    def transform_to_coordinate(self, ed, size=100, sigma=1.5):
        side_length = np.min(np.multiply((size / 2), ed.calibration))
        mask_for_sides = np.all(
            (np.abs(ed.coordinates[:, 0:2]) < side_length), axis=1
        )

        spot_coords = np.add(
            ed.calibrated_coordinates[mask_for_sides], size / 2
        ).astype(int)

        spot_intens = ed.intensities[mask_for_sides]
        spot_indice = ed.indices[mask_for_sides]

        df=pd.DataFrame(data={'x':spot_coords[:, 0],
                            'y':spot_coords[:, 1], 
                            'intensity':spot_intens,
                            'index_h':spot_indice[:,0],
                            'index_k':spot_indice[:,1],
                            'index_l':spot_indice[:,2]}
                        )
        return df

    def generate(self):
        ediff = DiffractionGenerator(accelerating_voltage=self.acceleration_voltage, 
                                     shape_factor_model=sinc)
        
        dp_info_list=[]
        dp_list=[]

        for dp_id, euler_angle in enumerate(self.angle_df.to_numpy()):
            ed = ediff.calculate_ed_data(structure=self.struct, 
                                         reciprocal_radius=self.reciprocal_radius,
                                         with_direct_beam=True,
                                         rotation=euler_angle,
                                         max_excitation_error=self.max_excitation_error,
                                         )
            ed.calibration = self.calibration
            single_dp_info = self.transform_to_coordinate(ed, 
                                                   size=self.pattern_size, 
                                                   sigma=self.sigma)
            single_dp_info.insert(0, "dp_id",dp_id)
            dp_info_list.append(single_dp_info)

            single_dp = ed.get_diffraction_pattern(size=self.pattern_size, 
                                                 sigma=self.sigma)
            dp_list.append(single_dp)

        # Concatenate dataframes in dp_info_list and dp in dp_list
        self.dp_infos = pd.concat(dp_info_list)
        self.dp_infos = self.dp_infos.reset_index()
        del self.dp_infos['index']

        self.dp_images = np.stack(dp_list)

        # Save to .hs file
        self.dp_dataset = pxm.signals.ElectronDiffraction2D(dp_list)
        self.dp_dataset.set_diffraction_calibration(self.calibration)

        print('The number of simulated patterns was {}.'.format(self.dp_images.shape[0]))
        return self.dp_images , self.dp_infos