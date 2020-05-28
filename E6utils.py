import numpy as np
import xarray as xr
import scipy.constants as const

hbar = const.hbar
c = const.c
ep0 = const.epsilon_0


def rot_mat(axis=(1, 0, 0), angle=0.0):
    """
    :param axis: axis of rotation (3D vector)
    :param angle: rotation angle specified in degrees
    :return mat: Matrix which implements rotation.
    """
    axis = np.array(axis)
    n = axis/np.linalg.norm(axis)
    ca = np.cos(np.radians(angle))
    sa = np.sin(np.radians(angle))
    mat = np.array(
        [[n[0] ** 2 * (1 - ca) + ca, n[0] * n[1] * (1 - ca) - n[2] * sa, n[0] * n[2] * (1 - ca) + n[1] * sa],
         [n[1] * n[0] * (1 - ca) + n[2] * sa, n[1] ** 2 * (1 - ca) + ca, n[1] * n[2] * (1 - ca) - n[0] * sa],
         [n[2] * n[0] * (1 - ca) - n[1] * sa, n[2] * n[1] * (1 - ca) + n[0] * sa, n[2] ** 2 * (1 - ca) + ca]])
    return mat


def translate_vec(vec=np.array([0, 0, 0]), trans_vec=np.array([0, 0, 0])):
    return vec + trans_vec


def transform_vec(vec=np.array([0, 0, 0]), mat=np.identity(3)):
    return np.matmul(mat, vec)


def matrix_rot_v_onto_w(v, w):
    # Given two vectors v and w this method returns a matrix which rotates vector v onto w.
    v = np.array(v)
    w = np.array(w)
    v = v/np.linalg.norm(v)
    w = w/np.linalg.norm(w)
    angle = float(np.degrees(np.arccos(np.dot(v, w))))
    axis = np.cross(v, w)
    if all(axis == np.array([0, 0, 0])):
        mat = np.identity(3)
    else:
        mat = rot_mat(axis=axis, angle=angle)
    return mat


def diff_arb(da, coord_list, *args, **kwargs):
    # da is an xarray. Coord list is list of subsequent coordinate derivatives to take
    # of the xarray
    new_da = da
    for i in coord_list:
        new_da = new_da.differentiate(i, *args, **kwargs)
    return new_da


def hessian(da, x0, y0, z0):
    ind_dict = {0: 'x', 1: 'y', 2: 'z'}
    hess = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            hess[i, j] = diff_arb(da, [ind_dict[i], ind_dict[j]]).sel(x=x0, y=y0, z=z0, method='nearest')
    return hess


def gaussian_1d(x, x0=0.0, sx=1.0, A=1.0, B=0.0):
    return A * np.exp(-(1/2)*((x-x0)/sx)**2) + B


def gaussian_2d(x, y, x0=0.0, y0=0.0, sx=1.0, sy=1.0, A=1.0, B=0.0, theta=0.0):
    rx = np.cos(np.radians(theta))*(x-x0) - np.sin(np.radians(theta))*(y-y0)
    ry = np.sin(np.radians(theta))*(x-x0) + np.cos(np.radians(theta))*(y-y0)
    return A * np.exp(-(1/2)*((rx/sx)**2 + ((ry/sy)**2))) + B


def parabolic_2d(x, y, x0=0.0, y0=0.0, parx=1.0, pary=1.0, A=1.0, B=0.0, theta=0.0):
    rx = np.cos(np.radians(theta)) * (x - x0) - np.sin(np.radians(theta)) * (y - y0)
    ry = np.sin(np.radians(theta)) * (x - x0) + np.cos(np.radians(theta)) * (y - y0)
    return A * np.maximum(1-(rx/parx)**2-(ry/pary)**2, 0)**1.5 + B


def bimodal_2d(x, y, x0=0.0, y0=0.0, sx=1.0, sy=1.0, parx=0.1, pary=0.1,
               peakG=1, peakP=1, B=0, theta=0):
    tot = B + gaussian_2d(x, y, x0, y0, sx, sy, peakG, 0, theta) \
          + parabolic_2d(x, y, x0, y0, parx, pary, peakP, 0, theta)
    return tot


def single_to_triple(x):
    try:
        len(x)
    except TypeError:
        x = (x, x, x)
    return x


def func3d_xr(f, x0=(-1,)*3, xf=(1,)*3, n_steps=(10,)*3):
    x0 = single_to_triple(x0)
    xf = single_to_triple(xf)
    n_steps = single_to_triple(n_steps)

    slicer = [slice(x0[d], xf[d], 1j*n_steps[d]) for d in range(3)]
    rvec = np.mgrid[slicer]
    field_array = np.vectorize(f)(rvec[0], rvec[1], rvec[2])
    x_coord = np.linspace(x0[0], xf[0], n_steps[0])
    y_coord = np.linspace(x0[1], xf[1], n_steps[1])
    z_coord = np.linspace(x0[2], xf[2], n_steps[2])
    field_xr = xr.DataArray(field_array,
                            coords=[
                                    ('x', x_coord),
                                    ('y', y_coord),
                                    ('z', z_coord)
                                    ])
    return field_xr


def resize_range(rmin=0, rmax=1, scale=1.1):
    cent = (rmin + rmax)/2
    half_rng = scale*(rmax - rmin)/2
    return [cent - half_rng, cent + half_rng]


def img_moments(img):
    rvec = np.indices(img.shape)
    tot = img.sum()
    if tot <= 0:
        raise ValueError('Integrated image intensity is negative, image may be too noisy. '
                         'Image statistics cannot be calculated.')
    x0 = np.sum(img*rvec[0])/tot
    y0 = np.sum(img*rvec[1])/tot
    varx = np.sum(img * (rvec[0] - x0)**2) / tot
    vary = np.sum(img * (rvec[1] - y0)**2) / tot
    if varx <= 0 or vary <= 0:
        raise ValueError('varx or vary is negative, image may be too noisy. '
                         'Image statistics cannot be calculated.')
    sx = np.sqrt(varx)
    sy = np.sqrt(vary)
    return x0, y0, sx, sy


def e_field_to_intensity(E):
    return (1/2)*c*ep0*E**2


def intensity_to_e_field(intensity):
    return np.sqrt(2*intensity/(c*ep0))
