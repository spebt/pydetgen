import pydetgen
import numpy as np
import matplotlib.pyplot as plt

# Get default configuration
config = pydetgen.default_config()

# Modify the configuration
# Generate new detector units geometry
det_pos = pydetgen.pattern.checkerboard((2.68, 1.68), (3.36, 3.36), (8, 32))
det_dims = np.array([3, 2, 1])
det_geoms = pydetgen.geometry.get_detector_units(0.475, det_pos.T, det_dims)
plate_width = 3.36 * 32
aperture_pitch = 5
aperture_width = 3
aperture_pos = np.arange(aperture_pitch / 2, plate_width, aperture_pitch)
aperture_pos = (plate_width - aperture_pos.shape[0] * 5) * 0.5 + aperture_pos
plate_geoms = pydetgen.geometry.get_plate(3.5, 0, plate_width, aperture_width, aperture_pos, 0, 1, 1)
config['detector']['detector geometry'] = np.append(plate_geoms, det_geoms, axis=0).tolist()
config['detector']['active geometry indices'] = (np.arange(8*16)+1).tolist()
config['FOV']['N voxels xyz'] = [180, 180, 1]
config['relation']['radial shift']['data'] = [100]
config['relation']['rotation']['data'] = [0,60,120,180,240,300]
# # Write the configuration to a yaml file
outfname = "panel_all.yaml"
pydetgen.yaml.write(config, outfname)
