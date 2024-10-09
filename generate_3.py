import pydetgen
import numpy as np

# Get default configuration
config = pydetgen.default_config()

# Modify the configuration
# Generate new detector units geometry
det_pos = pydetgen.pattern.checkerboard((2.68, 1.68), (3.36, 3.36), (7, 32))
outmost_x = np.ones(32) * (2.68 + 3.36 * 7)
outmost_y = 1.68 + np.arange(32) * 3.36
outmost_z = np.zeros(32)
outmost_det_pos = np.vstack((outmost_x, outmost_y, outmost_z))
det_dims = np.array([3, 3, 1])
det_pos = np.hstack((det_pos, outmost_det_pos))
sortarg = np.argsort(det_pos, axis=1)
print(sortarg.shape)
det_pos = det_pos[:, sortarg[0, :]]

det_geoms = pydetgen.geometry.get_detector_units(0.475, det_pos.T, det_dims)
plate_width = 3.36 * 32
aperture_pitch = 10
aperture_width = 2
N_apertures = int(np.floor((plate_width - aperture_width) / aperture_pitch)) + 1
plate_margin = (plate_width - (N_apertures - 1) * aperture_pitch - aperture_width) * 0.5
aperture_pos = (
    np.arange(N_apertures) * aperture_pitch + plate_margin + aperture_width * 0.5
)

plate_geoms = pydetgen.geometry.get_plate(
    3.5, 0, plate_width, aperture_width, aperture_pos, 0, 1, 1
)
config["detector"]["detector geometry"] = np.append(
    plate_geoms, det_geoms, axis=0
).tolist()
config["detector"]["active geometry indices"] = (np.arange(8 * 16) + 1).tolist()
config["FOV"]["N voxels xyz"] = [180, 180, 1]
config["relation"]["radial shift"]["data"] = [100]
config["relation"]["rotation"]["data"] = [0, 60, 120, 180, 240, 300]
# # Write the configuration to a yaml file
outfname = "six_panel_1z_full.yaml"
pydetgen.yaml.write(config, outfname)
