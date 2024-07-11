import numpy as np
import datetime


def get_plate_geoms_2d(mu: float, start_y: float, width_y: float, aperture_width: np.ndarray, aperture_pos: np.ndarray, start_x: float, thickness_x: float, height_z: float) -> np.ndarray:
    # apperture_pos is a 1-D Numpy.ndarray consisting of the y-coordinates of the aperture centers.
    # apperture_width is a 1-D Numpy.ndarray consisting of the aperture width. They can be different.
    n_aperture = aperture_pos.shape[0]
    ycoords = np.empty(n_aperture*2, dtype=np.float64)
    ycoords[::2] = - aperture_width*0.5 + aperture_pos
    ycoords[1::2] = aperture_width*0.5 + aperture_pos
    ycoords = np.append(np.insert(ycoords, 0, start_y), start_y+width_y)
    assert np.unique(
        ycoords).shape[0] == ycoords.shape[0], "Apertures overlapped, check the inputs!"
    geoms = np.empty((n_aperture+1, 8), dtype=np.float64)
    geoms[:, 0] = start_x
    geoms[:, 1] = start_x + thickness_x
    geoms[:, 4:6] = np.array([-height_z*0.5, height_z*0.5])
    geoms[:, 2] = ycoords[::2]
    geoms[:, 3] = ycoords[1::2]
    geoms[:, 6] = 0
    geoms[:, 7] = mu
    return geoms


def get_det_unit_geoms(mu: np.ndarray, pos: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """
    dims: can be of shape (3,) or (N points,3)
    """
    if pos.shape[0] != dims.shape[0] and dims.shape[0] != 3:
        raise Exception("Dimension input is in wrong shape.")
    indice = np.arange(pos.shape[0])+1
    geoms = np.empty((pos.shape[0], 8), dtype=np.float64)
    geoms[:, 6] = indice
    if pos.shape == dims.shape:
        geoms[:, :6:2] = pos[:] - dims[:]*0.5
        geoms[:, 1:6:2] = pos[:] + dims[:]*0.5
        # geoms[:, 1] = pos[:, 0] + dims[:, 0]*0.5
        # geoms[:, 2] = pos[:, 1] - dims[:, 1]*0.5
        # geoms[:, 3] = pos[:, 1] + dims[:, 1]*0.5
        # geoms[:, 4] = pos[:, 2] + dims[:, 2]*0.5
        # geoms[:, 5] = pos[:, 2] + dims[:, 2]*0.5
    if dims.shape == (3,):
        geoms[:, :6:2] = pos[:] - dims*0.5
        geoms[:, 1:6:2] = pos[:] + dims*0.5
    geoms[:, 7] = mu
    return geoms


def write_config_to_file(fname: str, plate_geoms, det_geoms, active_det_idx, det_nsubs, img_nsubs, nvx, mmpvx, dist, rotation_deg):
    active_det_idx_str = '['+', '.join(str(el) for el in active_det_idx)+']'
    indent_level_N_spaces = range(0, 10, 2)
    geom_lines_level = 3
    now_str = datetime.datetime.now().strftime('%x %X')
    plate_geoms_lines = []
    for geoms in plate_geoms:
        plate_geoms_lines.append('['+', '.join(str(el) for el in geoms)+']')
    plate_geoms_line_block = ''
    if len(plate_geoms_lines) != 0:
        plate_geoms_line_block = ' ' * indent_level_N_spaces[geom_lines_level]+(',\n'+' '*indent_level_N_spaces[geom_lines_level]
                                                                                ).join(el for el in plate_geoms_lines)+',\n'
    det_geoms_lines = []
    for geoms in det_geoms:
        det_geoms_lines.append('['+', '.join(str(el) for el in geoms)+']')

    det_geoms_line_block = ' '*indent_level_N_spaces[geom_lines_level] + (
        ',\n'+' '*indent_level_N_spaces[geom_lines_level]).join(el for el in det_geoms_lines)

    with open(fname, 'w') as fout:
        fout.write(
            '# This is an automatically generated systematic matrix configuration file in YAML format\n')
        fout.write('# Generated on: ' + now_str + '\n')
        fout.write('detector:\n')
        fout.write(' '*indent_level_N_spaces[1]+'detector geometry:\n')
        fout.write(' '*indent_level_N_spaces[2] +
                   '# detector geometry in millimeter.\n')
        fout.write(
            ' '*indent_level_N_spaces[2]+'# Defined in cuboids with x_0, x_1, y_0, y_1, z_0, z_1\n')
        fout.write(
            ' '*indent_level_N_spaces[2]+'# parameter 0 to 1: radial coordinates, x_0, x_1\n')
        fout.write(
            ' '*indent_level_N_spaces[2]+'# parameter 2 to 3: tangential coordiantes, y_0, y_1\n')
        fout.write(
            ' '*indent_level_N_spaces[2]+'# parameter 4 to 5: axial coordiantes, z_0, z_1\n')
        fout.write(' '*indent_level_N_spaces[2] +
                   '# # parameter 6: cuboid type identifier.\n')
        fout.write(
            ' '*indent_level_N_spaces[2]+'# 0 is non-detector, 1 and greater numbers are sequential indices for detector units.\n')
        fout.write(
            ' '*indent_level_N_spaces[2]+'# parameter 7: cuboid attenuation coefficient\n')
        fout.write(' '*indent_level_N_spaces[2]+'[\n')
        if len(plate_geoms_lines) != 0:
            fout.write(plate_geoms_line_block)
        fout.write(det_geoms_line_block)
        fout.write('\n'+' '*indent_level_N_spaces[2]+']\n')
        fout.write(' '*indent_level_N_spaces[1]+'N-subdivision xyz: [%d, %d, %d]' %
                   (det_nsubs[0], det_nsubs[1], det_nsubs[2])+'\n')
        fout.write(
            ' '*indent_level_N_spaces[1]+'active geometry indices: %s\n' % active_det_idx_str)
        fout.write(' '*indent_level_N_spaces[0]+'# Image space parameters\n')
        fout.write(' '*indent_level_N_spaces[0]+'image:\n')
        fout.write(
            ' '*indent_level_N_spaces[1]+'N-voxels xyz: [%d, %d, %d]\n' % (nvx[0], nvx[1], nvx[2]))
        fout.write(
            ' '*indent_level_N_spaces[1]+'mm-per-voxel xyz: [%d, %d, %d]\n' % (mmpvx[0], mmpvx[1], mmpvx[2]))
        fout.write(' '*indent_level_N_spaces[1]+'N-subdivision xyz: [%d, %d, %d]\n' % (
            img_nsubs[0], img_nsubs[1], img_nsubs[2]))
        fout.write(
            ' '*indent_level_N_spaces[0]+'# Image space to detector space relative positioning\n')
        fout.write(' '*indent_level_N_spaces[0]+'detector-to-image:\n')
        fout.write(
            ' '*indent_level_N_spaces[1]+'# detector front edge to FOV center distance in radial direction\n')
        fout.write(' '*indent_level_N_spaces[1] +
                   '# acceptable units are mm, cm, m\n')
        fout.write(
            ' '*indent_level_N_spaces[1]+'radial distance: %s mm\n' % str(dist))
        fout.write(
            ' '*indent_level_N_spaces[1]+'# rotation of the detector relative to the FOV in degrees\n')
        fout.write(
            ' '*indent_level_N_spaces[1]+'rotation: %s\n' % str(rotation_deg))
