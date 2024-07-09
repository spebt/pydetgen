import sys
sys.path.insert(0, '..')
import numpy as np
import pydetgen
import plotly.graph_objects as go



def cylinder(x0, y0, r, h, a=0, nt=100, nv=50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    theta = np.linspace(0, 2*np.pi, nt)
    v = np.linspace(a, a+h, nv)
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)+x0
    y = r*np.sin(theta)+y0
    z = v
    return x, y, z


def round_caps(x0, y0, r, h, a=0, nt=100, nv=50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    theta = np.linspace(0, 2*np.pi, nt)
    v = np.ones(nt)
    # theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)+x0
    y = r*np.sin(theta)+y0
    z1 = v*a
    z2 = v*(a+h)
    return x, y, z1, z2


data = []


focal_length = 10
plate_width = 50
aperture_width = 1
det_positions=np.array([[focal_length+0.5, plate_width*0.5, 0]])
det_sizes = np.array([3, 2, 1])

x_cylinder, y_cylinder, z_cylinder = cylinder(-20, 25, 15, 1, a=-0.5)


plate_geoms = pydetgen.get_plate_geoms_2d(10, 0, plate_width, aperture_width, np.array([plate_width*0.5]), 0, 1, 1)
det_geoms = pydetgen.get_det_unit_geoms(
    0.48, det_positions, det_sizes)

for geom in plate_geoms:
    data += [go.Mesh3d(x=np.tile(geom[0:2], 2), y=np.repeat(geom[2:4], 2),
                       z=np.full(4, geom[iz]), color='gray') for iz in [4, 5]]
    data += [go.Mesh3d(x=np.full(4, geom[ix]), y=np.repeat(geom[2:4], 2),
                       z=np.tile(geom[4:6], 2), color='gray', delaunayaxis='x') for ix in [0, 1]]
    data += [go.Mesh3d(x=np.repeat(geom[0:2], 2), y=np.full(4, geom[iy]),
                       z=np.tile(geom[4:6], 2), color='gray', delaunayaxis='y') for iy in [2, 3]]

for geom in det_geoms:
    data += [go.Mesh3d(x=np.tile(geom[0:2], 2), y=np.repeat(geom[2:4], 2),
                       z=np.full(4, geom[iz]), color='orange') for iz in [4, 5]]
    data += [go.Mesh3d(x=np.full(4, geom[ix]), y=np.repeat(geom[2:4], 2),
                       z=np.tile(geom[4:6], 2), color='orange', delaunayaxis='x') for ix in [0, 1]]
    data += [go.Mesh3d(x=np.repeat(geom[0:2], 2), y=np.full(4, geom[iy]),
                       z=np.tile(geom[4:6], 2), color='orange', delaunayaxis='y') for iy in [2, 3]]


colorscale = [[0, 'blue'],
              [1, 'lightblue']]
cyl1 = [go.Surface(x=x_cylinder, y=y_cylinder, z=z_cylinder,
                   colorscale=colorscale,
                   showscale=False,
                   opacity=0.3)]

cap_x, cap_y, cap_z1, cap_z2 = round_caps(-20, 25, 15, 1, a=-0.5)
print(cap_z2.shape)
cap_top = [go.Mesh3d(x=cap_x, y=cap_y, z=cap_z1,
                     color='lightblue',
                     showscale=False,
                     opacity=0.3)]
cap_bot = [go.Mesh3d(x=cap_x, y=cap_y, z=cap_z2,
                     color='lightblue',
                     showscale=False, opacity=0.5)]


fig = go.Figure(data=data+cyl1+cap_top+cap_bot)
# aspectratio_x=1.25,aspectratio_y=1.25,aspectratio_z=1.25,
fig.update_scenes(aspectmode='data')
name = 'default'
# Default parameters which are used when `layout.scene.camera` is not provided
camera = dict(
    eye=dict(x=4.25, y=2.25, z=3.5)
)

fig.update_layout(scene_camera=camera, title=name)
fig.show()
