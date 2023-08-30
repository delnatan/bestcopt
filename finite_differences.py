import numpy as np


def compute_1st_order_filters_1d(n):
    i2pi = 2.0 * np.pi * 1j
    kx = np.fft.rfftfreq(n)
    Dx = 1 - np.exp(-i2pi * kx)
    DtD = np.conj(Dx) * Dx
    return [Dx], 1.0 + DtD


def compute_1st_order_filters_2d(n):
    i2pi = 2.0 * np.pi * 1j
    fy = np.fft.fftfreq(n[0])
    fx = np.fft.rfftfreq(n[1])
    ky, kx = np.meshgrid(fy, fx, indexing='ij')
    Dx = 1 - np.exp(-i2pi * kx)
    Dy = 1 - np.exp(-i2pi * ky)
    DtD = np.conj(Dx) * Dx + np.conj(Dy) * Dy
    return [Dy, Dx], 1.0 + DtD


def compute_1st_order_filters_3d(n):
    i2pi = 2.0 * np.pi * 1j
    fz = np.fft.fftfreq(n[0])
    fy = np.fft.fftfreq(n[1])
    fx = np.fft.rfftfreq(n[2])
    kz, ky, kx = np.meshgrid(fz, fy, fx, indexing='ij')
    Dz = 1 - np.exp(-i2pi * kz)
    Dy = 1 - np.exp(-i2pi * ky)
    Dx = 1 - np.exp(-i2pi * kx)
    DtD = np.conj(Dz) * Dz + \
        np.conj(Dy) * Dy + \
        np.conj(Dx) * Dx
    return [Dz, Dy, Dx], 1.0 + DtD


def compute_2nd_order_filters_1d(n):
    """compute second-order finite difference filters
    """
    i2pi = 2.0 * np.pi * 1j
    # frequency space:
    kx = np.fft.rfftfreq(n)
    filt = np.exp(-i2pi * kx) - 2 + np.exp(i2pi * kx)
    DtD = np.conj(filt) * filt
    return [filt], 1.0 + DtD


def compute_2nd_order_filters_2d(n):
    i2pi = 2.0 * np.pi * 1j
    fy = np.fft.fftfreq(n[0])
    fx = np.fft.rfftfreq(n[1])
    ky, kx = np.meshgrid(fy, fx, indexing='ij')

    Dxx = np.exp(-i2pi * kx) - 2 + np.exp(i2pi * kx)
    Dyy = np.exp(-i2pi * ky) - 2 + np.exp(i2pi * ky)
    Dyx = (
        1 - np.exp(-i2pi * ky) - np.exp(-i2pi * kx)
        + np.exp(-i2pi * (ky + kx))
    )

    # compute Gram matrix D^t.D
    DtD = np.conj(Dxx) * Dxx + \
        np.conj(Dyy) * Dyy + \
        2 * np.conj(Dyx) * Dyx

    return [Dyy, Dxx, 2 * Dyx], 1.0 + DtD


def compute_2nd_order_filters_3d(n):
    i2pi = 2.0 * np.pi * 1j
    fz = np.fft.fftfreq(n[0])
    fy = np.fft.fftfreq(n[1])
    fx = np.fft.rfftfreq(n[2])

    kz, ky, kx = np.meshgrid(fz, fy, fx, indexing='ij')

    Dxx = np.exp(-i2pi * kx) - 2 + np.exp(i2pi * kx)
    Dyy = np.exp(-i2pi * ky) - 2 + np.exp(i2pi * ky)
    Dzz = np.exp(-i2pi * kz) - 2 + np.exp(i2pi * kz)
    Dyx = (
        1 - np.exp(-i2pi * ky) - np.exp(-i2pi * kx)
        + np.exp(-i2pi * (ky + kx))
    )
    Dyz = (
        1 - np.exp(-i2pi * ky) - np.exp(-i2pi * kz)
        + np.exp(-i2pi * (ky + kz))
    )
    Dxz = (
        1 - np.exp(-i2pi * kz) - np.exp(-i2pi * kx)
        + np.exp(-i2pi * (kz + kx))
    )

    DtD = np.conj(Dxx) * Dxx + \
        np.conj(Dyy) * Dyy + \
        np.conj(Dzz) * Dzz + \
        2 * np.conj(Dyx) * Dyx + \
        2 * np.conj(Dyz) * Dyz + \
        2 * np.conj(Dxz) * Dxz

    return [Dzz, Dyy, Dxx, 2 * Dyz, 2 * Dxz, 2 * Dyx], 1.0 + DtD


def compute_2nd_order_filters(shape, ndim):
    if ndim==1:
        return compute_2nd_order_filters_1d(shape)
    elif ndim==2:
        return compute_2nd_order_filters_2d(shape)
    elif ndim==3:
        return compute_2nd_order_filters_3d(shape)