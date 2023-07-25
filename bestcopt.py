import numpy as np
from scipy.ndimage import map_coordinates


def compute_2d_diff_filters(shape):
    """
    Compute the stacked operator D = [Dxx, Dyy, sqrt(2)*Dxy]

    """
    assert len(shape) == 2, "`shape` must have 2 elements"

    fy = np.fft.fftfreq(shape[0])
    fx = np.fft.rfftfreq(shape[1])
    ky, kx = np.meshgrid(fy, fx, indexing='ij')

    sqrt2 = np.sqrt(2.0)
    twopi = 2.0 * np.pi

    Dxx = np.exp(-twopi * 1j * kx) - 2 + np.exp(twopi * 1j * kx)
    Dyy = np.exp(-twopi * 1j * ky) - 2 + np.exp(twopi * 1j * ky)
    Dyx = (
        1 - np.exp(-twopi * 1j * ky) - np.exp(-twopi * 1j * kx)
        + np.exp(-twopi * 1j * (ky + kx))
    )

    return [Dyy, Dxx, sqrt2 * Dyx]


def compute_3d_diff_filters(shape, lateral_to_axial_ratio):
    """
    Compute the stacked operator
    D = [Dxx, Dyy, Dzz, sqrt(2) * Dyx, sqrt(2) * Dyz, sqrt(2) * Dyx]
    """
    d = lateral_to_axial_ratio

    fz = np.fft.fftfreq(shape[0])
    fy = np.fft.fftfreq(shape[1])
    fx = np.fft.fftfreq(shape[2])

    kz, ky, kx = np.meshgrid(fz, fy, fx, indexing="ij")

    sqrt2 = np.sqrt(2.0)
    twopi = 2.0 * np.pi

    Dxx = np.exp(-twopi * 1j * kx) - 2 + np.exp(twopi * 1j * kx)
    Dyy = np.exp(-twopi * 1j * ky) - 2 + np.exp(twopi * 1j * ky)
    Dzz = d**2 * np.exp(-twopi * 1j * kz) - 2 * np.exp(twopi * 1j * kz)
    Dyx = (
        1 - np.exp(-twopi * 1j * ky) - np.exp(-twopi * 1j * kx)
        + np.exp(-twopi * 1j * (ky + kx))
    )
    Dxz = sqrt2 * d * (
        1 - np.exp(-twopi * 1j * kx) - np.exp(-twopi * 1j * kz)
        + np.exp(-twopi * 1j * (kx + kz))
    )
    Dyz = sqrt2 * d * (
        1 - np.exp(-twopi * 1j * ky) - np.exp(-twopi * 1j * kz)
        + np.exp(-twopi * 1j * (ky + kz))
    )

    return [Dyy, Dxx, Dzz, Dyz, Dxz, Dyx]


def estimate_baseline_2d(b, lam=0.1, max_iter=500, pad=10, rel_tol=1e-3):

    prob_shape = [s + 2 * pad for s in b.shape]

    ft_D = compute_2d_diff_filters(prob_shape)
    ft_DtD = sum([np.conj(f) * f for f in ft_D])

    relscores = []

    # pad input image
    bpad = np.zeros(prob_shape)
    bpad[pad:-pad, pad:-pad] = b
    mask_in = np.zeros(prob_shape, dtype=bool)
    mask_in[pad:-pad, pad:-pad] = True
    mask_out = ~mask_in

    # pre-allocate (auxilliary) variables
    x = np.zeros(prob_shape)
    z1 = np.zeros(prob_shape)
    z2 = np.zeros([3,] + prob_shape)
    u1 = np.zeros(prob_shape)
    u2 = np.zeros([3,] + prob_shape)

    rho = 8000.0 * lam / b.max()

    print(f"rho = {rho:10.4f}")

    for i in range(max_iter):

        ft_v = np.fft.rfft2(z2 - u2)
        Dt_v = sum(
            [np.fft.irfft2(np.conj(f) * ft_v[i], s=prob_shape) for i,f in enumerate(ft_D)]
        )
        ft_rhs = np.fft.rfft2(bpad - z1 + u1 + Dt_v)

        # solve for x (x-update)
        xold = x
        x = np.fft.irfft2(ft_rhs / (1.0 + ft_DtD), s=prob_shape)
        dx = np.linalg.norm(x - xold) / max(np.linalg.norm(xold), 1e-7)
        relscores.append(dx)

        if dx < rel_tol:
            break

        # z1 update
        v = -x + bpad + u1
        z1 = np.sign(v) * np.maximum(np.abs(v) - 1 / rho, 0)
        z1 = mask_in * np.maximum(z1, 0) + mask_out * v

        # z2 update
        ft_x = np.fft.rfft2(x)
        Dx = np.stack([np.fft.irfft2(f * ft_x, s=prob_shape) for f in ft_D])
        v = Dx - u2
        z2 = mask_in * ((rho * v) / (rho - 2 * lam)) + mask_out * v

        # u update
        u1 = u1 + (bpad - x - z1)
        u2 = u2 + (Dx - z2)

        print(f"\riteration = {i+1:d}; |dx| = {dx:12.4E}", end="")

    print("\n")

    return x[pad:-pad, pad:-pad]


def estimate_baseline_3d(b, lam=0.1, max_iter=500, pad=10, rel_tol=1e-3):

    prob_shape = [s + 2 * pad for s in b.shape]

    ft_D = compute_2d_diff_filters(prob_shape)
    ft_DtD = sum([np.conj(f) * f for f in ft_D])

    relscores = []

    # pad input image
    bpad = np.zeros(prob_shape)
    bpad[pad:-pad, pad:-pad, pad:-pad] = b
    mask_in = np.zeros(prob_shape, dtype=bool)
    mask_in[pad:-pad, pad:-pad, pad:-pad] = True
    mask_out = ~mask_in

    # pre-allocate (auxilliary) variables
    x = np.zeros(prob_shape)
    z1 = np.zeros(prob_shape)
    z2 = np.zeros([6,] + prob_shape)
    u1 = np.zeros(prob_shape)
    u2 = np.zeros([6,] + prob_shape)

    rho = 8000.0 * lam / b.max()

    for i in range(max_iter):

        ft_v = np.fft.rfftn(z2 - u2)
        Dt_v = sum(
            [np.fft.irfftn(np.conj(f) * ft_v[i], s=prob_shape) for i,f in enumerate(ft_D)]
        )
        ft_rhs = np.fft.rfftn(bpad - z1 + u1 + Dt_v)

        # solve for x (x-update)
        xold = x
        x = np.fft.irfftn(ft_rhs / (1 + ft_DtD), s=prob_shape)
        dx = np.linalg.norm(x - xold) / max(np.linalg.norm(xold), 1e-7)
        relscores.append(dx)

        if dx < rel_tol:
            break

        # z1 update
        v = -x + bpad + u1
        z1 = np.sign(v) * np.maximum(np.abs(v) - 1 / rho, 0)
        z1 = mask_in * np.maximum(z1, 0) + mask_out * v

        # z2 update
        ft_x = np.fft.rfft2(x)
        Dx = np.stack([np.fft.irfftn(f * ft_x, s=prob_shape) for f in ft_D])
        v = Dx - u2
        z2 = mask_in * ((rho * v) / (rho - 2 * lam)) + mask_out * v

        # u update
        u1 = u1 + (bpad - x - z1)
        u2 = u2 + (Dx - z2)

        print(f"\riteration = {i+1:d}; |dx| = {dx:12.4E}", end="")

    print("\n")

    return x[pad:-pad, pad:-pad, pad:-pad]


def line_profile(img, p1, p2):
    npts = int(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[0]) ** 2))
    ypts = np.linspace(p1[0], p2[0], num=npts)
    xpts = np.linspace(p1[1], p2[1], num=npts)
    yxpts = np.column_stack((ypts, xpts)).T
    return map_coordinates(img, yxpts, order=2)
