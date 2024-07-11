import plotly.graph_objects as go
import pydetgen
import numpy as np
import sys
import os.path
import re

fname = ''
if len(sys.argv) == 2:
    fname = sys.argv[1]
else:
    stdin_lines = []
    for line in sys.stdin:
        stdin_lines.append(line)
    fname = stdin_lines[0].rstrip()
assert os.path.exists(fname), 'Invalid file name or path!'
imgFname=re.match('^(.+).yml',fname).group(1)+'.png'

config = pydetgen.get_config(fname)
allgeoms = config['det geoms']
plate_geoms = allgeoms[allgeoms[:, 6] == 0]
det_geoms = allgeoms[allgeoms[:, 6] != 0]
det_dims = np.max(allgeoms[:, 1:6:2], axis=0) - \
    np.min(allgeoms[:, 0:6:2], axis=0)
img_dims = config['img nvx']*config['mmpvx']

fig = pydetgen.draw_det_geoms(
    plate_geoms, det_geoms, -config['dist'], det_dims[1]*0.5, 0, np.min(img_dims[0:2])*0.5, img_dims[2])


name = fname
camera = dict(
    eye=dict(x=0, y=0, z=15),
    # projection=dict(type='orthographic'),
    # eye=dict(x=0., y=0., z=2)

)

fig.update_layout(scene_camera=camera, title=name)
fig.update_scenes(aspectmode='data')
fig.update_layout(scene=dict(zaxis_visible=False))
fig.update_layout(
    autosize=False,
    width=1000,
    height=2000,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=4
    ),
)
# fig.show()
fig.write_image(imgFname)
