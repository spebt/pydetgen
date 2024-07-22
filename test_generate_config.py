import pydetgen
import numpy as np
import sys
import os.path
import datetime
fname = []
fname_new = 'test_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.yml'
if len(sys.argv) != 2:
    print('%s' % fname_new)
    fname = fname_new
else:
    fname = sys.argv[1]
    if os.path.exists(fname):
        print('%s' % (fname_new))
        fname = fname_new
    else:
        print('%s' % fname)

focal_length = 50
plate_width = 100
aperture_width = 1
n_det = 1
det_sizes = np.array([3, 2, 1])
det_interval_y = 0.5
pos_x = np.full(n_det, focal_length+0.5)
pos_z = np.zeros(n_det)
pos_y = np.hstack((-np.arange(1, n_det//2+1) *
                  (det_sizes[1]+det_interval_y), 0, np.arange(1, n_det//2+1)*(det_sizes[1]+det_interval_y)))+plate_width*0.5
det_positions = np.vstack((pos_x, pos_y, pos_z)).T

# pos_x2 = np.full(n_det, focal_length*0.5+0.5)
# pos_y2 = np.hstack((-np.arange(1, n_det//4+1) *
#                     (5), 0, np.arange(1, n_det//4+1)*(5)))+plate_width*0.5
# det_positions_2 = np.vstack((pos_x2[:n_det//2], pos_y2, pos_z[:n_det//2])).T
# det_positions = np.concatenate((det_positions, det_positions_2), axis=0)

det_nsubs = np.array([4, 4, 1])
img_nsubs = np.array([4, 4, 1])
nvx = np.array([100, 100, 1])
mmpvx = np.array([1, 1, 1])
dist = 55
rotation_deg = 0

# aperture_pos_y = np.array([-5,5])+plate_width*0.5
aperture_pos_y = np.array([0])+plate_width*0.5
plate_geoms = pydetgen.get_plate_geoms_2d(
    10, 0, plate_width, aperture_width, aperture_pos_y, 0, 1, 1)
det_geoms = pydetgen.get_det_unit_geoms(
    0.48, det_positions, det_sizes)
active_det_idx = [1]

pydetgen.write_config_to_file(fname, plate_geoms, det_geoms, active_det_idx,
                              det_nsubs, img_nsubs, nvx, mmpvx, dist, rotation_deg)
