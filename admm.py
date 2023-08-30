import numpy as np
import finite_differences


def remove_pad(padded_array, pad):
    """return unpadded n-dimensional array"""
    if isinstance(pad, int):
        pad = (pad,) * padded_array.ndim
    slices = [slice(p, -p if p != 0 else None) for p in pad]
    return padded_array[tuple(slices)]


def get_pad_slices(pad, ndim):
    """return slices for 'inside' images"""
    if isinstance(pad, int):
        pad = (pad,) * ndim
    slices = [slice(p, -p if p != 0 else None) for p in pad]
    return tuple(slices)


class State:
    def __init__(self, data, padding=10):
        self.data = data
        self.ndim = data.ndim
        self.pad = padding
        self.scale = data.max()
        self.b = (
            np.pad(self.data, self.pad, mode="constant", constant_values=0)
            / self.scale
        )
        self.padded_shape = self.b.shape
        (
            self.filters,
            self.ft_denominator,
        ) = finite_differences.compute_2nd_order_filters(
            self.padded_shape, self.ndim
        )

        inside_slices = get_pad_slices(self.pad, self.ndim)
        self.mask_in = np.zeros(self.padded_shape, dtype=bool)
        self.mask_in[inside_slices] = True
        self.mask_out = ~self.mask_in
        self.ft_axes = tuple([i + 1 for i in range(self.ndim)])
        self.initialize_variables()

    def initialize_variables(self):
        self.x = np.zeros(self.padded_shape)
        self.z1 = np.zeros(self.padded_shape)
        if self.ndim == 1:
            z2shape = self.padded_shape
        elif 1 < self.ndim <= 3:
            z2shape = (3 * (self.ndim - 1),) + self.padded_shape
        self.z2 = np.zeros(z2shape)
        self.u1 = np.zeros(self.padded_shape)
        self.u2 = np.zeros(z2shape)
        self.Dx = np.zeros(self.padded_shape)
        self.dz1 = np.zeros(self.padded_shape)
        self.dz2 = np.zeros(z2shape)


def update_x(s):
    if s.ndim == 1:
        ft_wrk = np.conj(s.filters[0]) * np.fft.rfft(s.z2 - s.u2)
        ft_numerator = np.fft.rfft(-s.z1 + s.b + s.u1) + ft_wrk
        s.x = np.fft.irfft(ft_numerator / s.ft_denominator, n=s.padded_shape[0])
    elif 1 < s.ndim <= 3:
        ft_wrk = np.fft.rfftn(s.z2 - s.u2, axes=s.ft_axes)
        ft_wrk = sum([np.conj(f) * ft_wrk[i] for i, f in enumerate(s.filters)])
        ft_numerator = np.fft.rfftn(-s.z1 + s.b + s.u1) + ft_wrk
        s.x = np.fft.irfftn(ft_numerator / s.ft_denominator, s=s.padded_shape)


def update_z1(s, rho):
    v = -s.x + s.b + s.u1
    z1prev = s.z1
    s.z1 = s.mask_in * np.maximum(v - 1 / rho, 0) + s.mask_out * v
    s.dz1 = s.z1 - z1prev


def update_z2(s, rho, lam):
    if s.ndim == 1:
        ft_wrk = np.fft.rfft(s.x)
        s.Dx = np.fft.irfft(s.filters[0] * ft_wrk, n=s.padded_shape[0])
    elif 1 < s.ndim <= 3:
        ft_wrk = np.fft.rfftn(s.x)
        s.Dx = np.stack(
            [np.fft.irfftn(f * ft_wrk, s=s.padded_shape) for f in s.filters]
        )
    z2prev = s.z2
    s.z2 = rho * (s.Dx + s.u2) / (2 * lam + rho)
    s.dz2 = s.z2 - z2prev


def update_residuals(s):
    s.u1 += -s.x + s.b - s.z1
    s.u2 += s.Dx - s.z2


def compute_primal_residual(s):
    res1 = -s.x - s.z1 + s.b
    res2 = s.Dx - s.z2
    if s.ndim == 1:
        return np.linalg.norm([res1, res2])
    elif 1 < s.ndim <= 3:
        return np.linalg.norm(np.concatenate((res1[None, ...], res2), axis=0))


def compute_dual_residual(s, rho):
    """s = ρ⋅Aᵀ⋅B(z - z_prev)
    A = K
    B = -I
    s = -ρ⋅Kᵀ⋅(z - z_prev)
    s1 = ρ⋅(z1 - z1_prev)
    s2 = -ρ⋅Dᵀ⋅(z2 - z2_prev)
    """

    # compute Dᵀ⋅Δz2
    if s.ndim == 1:
        Dt_dz2 = np.fft.irfft(
            np.conj(s.filters[0]) * np.fft.rfft(s.dz2), n=s.padded_shape[0]
        )
        return np.linalg.norm([rho * s.dz1, -rho * Dt_dz2])
    elif 1 < s.ndim <= 3:
        ft_dz2 = np.fft.rfftn(s.dz2, axes=s.ft_axes)
        Dt_dz2 = sum(
            [
                np.fft.irfftn(np.conj(f) * ft_dz2[i], s=s.padded_shape)
                for i, f in enumerate(s.filters)
            ]
        )
        return np.linalg.norm([rho * s.dz1, -rho * Dt_dz2])
