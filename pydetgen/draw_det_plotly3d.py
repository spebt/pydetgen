import numpy as np
import plotly.graph_objects as go


def __get_cylinder(x0, y0, z0, r, h,  nt=100, nv=50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    theta = np.linspace(0, 2*np.pi, nt)
    v = np.linspace(z0-h*0.5, z0+h*0.5, nv)
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)+x0
    y = r*np.sin(theta)+y0
    z = v
    return x, y, z


def __get_round_caps(x0, y0, z0, r, h, nt=100, nv=50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    theta = np.linspace(0, 2*np.pi, nt)
    v = np.ones(nt)
    # theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)+x0
    y = r*np.sin(theta)+y0
    z1 = v*(z0-h*0.5)
    z2 = v*(z0+h*0.5)
    return x, y, z1, z2


def draw_det_geoms(plate_geoms, det_geoms, fov_x0, fov_y0, fov_z0, fov_r, fov_h):
    data = []
    x_cylinder, y_cylinder, z_cylinder = __get_cylinder(
        fov_x0, fov_y0, fov_z0, fov_r, fov_h)
    cap_x, cap_y, cap_z1, cap_z2 = __get_round_caps(
        fov_x0, fov_y0, fov_z0, fov_r, fov_h)
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

    # print(cap_z2.shape)
    cap_top = [go.Mesh3d(x=cap_x, y=cap_y, z=cap_z1,
                         color='lightblue',
                         showscale=False,
                         opacity=0.3)]
    cap_bot = [go.Mesh3d(x=cap_x, y=cap_y, z=cap_z2,
                         color='lightblue',
                         showscale=False, opacity=0.5)]
    fig = go.Figure(data=data+cyl1+cap_top+cap_bot)
    fig.add_trace(go.Scatter3d(
        x=[fov_x0], y=[fov_y0], z=[fov_z0], mode="text", name='Name', text=['FOV'], textfont=dict(
            family="sans serif",
            size=36,
            color="black"
        )
    ))
    # aspectratio_x=1.25,aspectratio_y=1.25,aspectratio_z=1.25,
    fig.update_scenes(yaxis_ticksuffix=' mm',
                      xaxis_ticksuffix=' mm',
                      zaxis_ticksuffix=' mm')

    return fig
