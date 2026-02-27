"""Spectral analysis tools for VoxelCAD SDF volumes.

Provides radial power spectrum computation, bandwidth estimation,
cutoff recommendation, and visualization functions for understanding
the frequency content of signed distance fields.

Frequency units are cycles/voxel throughout (Nyquist = 0.5).
"""

import logging

import numpy as np
from scipy.fft import rfftn, fftfreq, rfftfreq
from scipy.stats import binned_statistic

LOGGER = logging.getLogger(__name__)


def radial_power_spectrum(volume, n_bins=None):
    """Compute radially-averaged power spectrum of a 3D volume.

    Parameters
    ----------
    volume : np.ndarray
        3D float array (SDF, binary voxels, etc.)
    n_bins : int, optional
        Number of radial frequency bins. Defaults to half
        the smallest grid dimension.

    Returns
    -------
    freq_bins : np.ndarray
        1D array of frequency bin centers (cycles/voxel).
    power : np.ndarray
        1D array of mean power per radial shell.
    """
    rx, ry, rz = volume.shape
    if n_bins is None:
        n_bins = min(rx, ry, rz) * 2

    F = rfftn(volume)
    P = np.abs(F) ** 2
    del F

    fx = fftfreq(rx)
    fy = fftfreq(ry)
    fz = rfftfreq(rz)
    FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
    freq_mag = np.sqrt(FX**2 + FY**2 + FZ**2)
    del FX, FY, FZ

    nyquist = 0.5 * np.sqrt(3.0)  # max radial freq in 3D
    bin_edges = np.linspace(0, nyquist, n_bins + 1)

    result = binned_statistic(
        freq_mag.ravel(), P.ravel(),
        statistic='mean', bins=bin_edges,
    )
    freq_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    power = np.nan_to_num(result.statistic, nan=0.0)

    return freq_bins, power


def estimate_bandwidth(freq_bins, power, energy_fraction=0.99):
    """Find frequency containing a fraction of total spectral energy.

    Parameters
    ----------
    freq_bins : np.ndarray
        Frequency bin centers from ``radial_power_spectrum``.
    power : np.ndarray
        Power per radial shell.
    energy_fraction : float
        Fraction of total energy (0 to 1). Default 0.99.

    Returns
    -------
    float
        Frequency (cycles/voxel) below which ``energy_fraction``
        of total power resides.
    """
    cumulative = np.cumsum(power)
    if cumulative[-1] == 0:
        return 0.0
    idx = np.searchsorted(cumulative, cumulative[-1] * energy_fraction)
    return freq_bins[min(idx, len(freq_bins) - 1)]


def safe_stride(bandwidth, safety_margin=1.2):
    """Compute maximum safe subsampling stride from bandwidth.

    After striding by S, Nyquist drops to 0.5/S. This must exceed
    the geometry bandwidth (with safety margin) to avoid aliasing.

    Parameters
    ----------
    bandwidth : float
        Geometry bandwidth in cycles/voxel.
    safety_margin : float
        Multiply bandwidth by this before computing stride. Default 1.2.

    Returns
    -------
    int
        Maximum safe stride (>= 1).
    """
    effective = bandwidth * safety_margin
    if effective <= 0:
        return 1
    return max(1, int(np.floor(0.5 / effective)))


def recommend_cutoff(freq_bins, power, method='energy'):
    """Suggest Butterworth cutoff based on spectral analysis.

    Parameters
    ----------
    freq_bins : np.ndarray
        Frequency bin centers.
    power : np.ndarray
        Power per radial shell.
    method : str
        ``'energy'`` — frequency at 99% cumulative energy (conservative).
        ``'knee'`` — inflection point in log-power curve.

    Returns
    -------
    float
        Recommended cutoff frequency (cycles/voxel).
    """
    if method == 'energy':
        bw = estimate_bandwidth(freq_bins, power, energy_fraction=0.99)
        return min(bw * 1.5, 0.45)
    elif method == 'knee':
        log_power = np.log10(power + 1e-30)
        d2 = np.gradient(np.gradient(log_power, freq_bins), freq_bins)
        # Knee = maximum curvature (most negative second derivative)
        # Search only in the mid-frequency range
        mask = (freq_bins > 0.02) & (freq_bins < 0.45)
        if not mask.any():
            return 0.25
        candidates = np.where(mask)[0]
        knee_idx = candidates[np.argmin(d2[mask])]
        return float(freq_bins[knee_idx])
    else:
        raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Visualization functions (matplotlib, Agg-compatible)
# ---------------------------------------------------------------------------

def plot_radial_spectrum(freq_bins, power, cutoff=None, bandwidth=None,
                         ax=None, title=None):
    """Log-log radial power spectrum with diagnostic annotations.

    Parameters
    ----------
    freq_bins, power : np.ndarray
        From ``radial_power_spectrum``.
    cutoff : float, optional
        Butterworth cutoff to mark (red dashed).
    bandwidth : float, optional
        Estimated bandwidth to mark (green dashed).
    ax : matplotlib Axes, optional
        Axes to plot on. Created if None.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    mask = power > 0
    ax.semilogy(freq_bins[mask], power[mask], 'b-', linewidth=1.5,
                label='Radial power')
    ax.axvline(0.5, color='0.5', linestyle=':', linewidth=1,
               label='Nyquist (0.5 cyc/vox)')
    if cutoff is not None:
        ax.axvline(cutoff, color='r', linestyle='--', linewidth=1.5,
                   label=f'Butterworth cutoff ({cutoff})')
    if bandwidth is not None:
        ax.axvline(bandwidth, color='g', linestyle='--', linewidth=1.5,
                   label=f'99% bandwidth ({bandwidth:.3f})')
    ax.set_xlabel('Frequency (cycles/voxel)')
    ax.set_ylabel('Mean Power')
    ax.set_title(title or 'Radial Power Spectrum')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.55)
    return fig


def plot_filter_effect(sdf_raw, sdf_filtered, cutoff=0.25,
                       n_bins=None, ax=None, title=None):
    """Overlay raw vs Butterworth-filtered power spectra.

    Parameters
    ----------
    sdf_raw, sdf_filtered : np.ndarray
        3D SDF arrays before and after filtering.
    cutoff : float
        Butterworth cutoff used.
    n_bins : int, optional
        Radial bins.
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    f_raw, p_raw = radial_power_spectrum(sdf_raw, n_bins=n_bins)
    f_filt, p_filt = radial_power_spectrum(sdf_filtered, n_bins=n_bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    m_raw = p_raw > 0
    m_filt = p_filt > 0
    ax.semilogy(f_raw[m_raw], p_raw[m_raw], 'b-', linewidth=1.5,
                label='Raw SDF', alpha=0.8)
    ax.semilogy(f_filt[m_filt], p_filt[m_filt], 'C1-', linewidth=1.5,
                label='After Butterworth', alpha=0.8)
    ax.axvline(cutoff, color='r', linestyle='--', linewidth=1.5,
               label=f'Cutoff ({cutoff})')
    ax.axvline(0.5, color='0.5', linestyle=':', linewidth=1,
               label='Nyquist')
    ax.set_xlabel('Frequency (cycles/voxel)')
    ax.set_ylabel('Mean Power')
    ax.set_title(title or f'Filter Effect (cutoff={cutoff})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.55)
    return fig


def plot_spectrum_slices(volume, log_scale=True, ax=None, title_prefix=None):
    """Show XY, XZ, YZ central slices through 3D power spectrum.

    Parameters
    ----------
    volume : np.ndarray
        3D float array.
    log_scale : bool
        If True, show log10(power).
    ax : array of 3 Axes, optional
    title_prefix : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    F = rfftn(volume)
    P = np.abs(F) ** 2
    del F

    rx, ry, rz_half = P.shape
    # Central slices (DC is at index 0 for fftfreq)
    slice_xy = P[:, :, 0]  # fz=0 plane
    slice_xz = P[:, 0, :]  # fy=0 plane
    slice_yz = P[0, :, :]  # fx=0 plane

    slices = [slice_xy, slice_xz, slice_yz]
    labels = ['XY (fz=0)', 'XZ (fy=0)', 'YZ (fx=0)']

    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        axes = ax
        fig = axes[0].figure

    for a, s, lbl in zip(axes, slices, labels):
        data = np.log10(np.fft.fftshift(s, axes=0) + 1e-30) if log_scale else np.fft.fftshift(s, axes=0)
        im = a.imshow(data, aspect='auto', cmap='viridis', origin='lower')
        a.set_title(f'{title_prefix + " " if title_prefix else ""}{lbl}')
        a.set_xlabel('Freq axis 2')
        a.set_ylabel('Freq axis 1')
        fig.colorbar(im, ax=a, shrink=0.8,
                     label='log10(Power)' if log_scale else 'Power')
    fig.tight_layout()
    return fig


def plot_cumulative_energy(freq_bins, power, markers=None, ax=None,
                           title=None):
    """Cumulative energy fraction vs frequency.

    Parameters
    ----------
    freq_bins, power : np.ndarray
        From ``radial_power_spectrum``.
    markers : dict, optional
        ``{label: frequency}`` pairs to annotate on the curve.
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    cumulative = np.cumsum(power)
    total = cumulative[-1] if cumulative[-1] > 0 else 1.0
    frac = cumulative / total

    ax.plot(freq_bins, frac, 'b-', linewidth=2)
    ax.axhline(0.99, color='0.4', linestyle=':', linewidth=1,
               label='99% energy')
    ax.axhline(0.95, color='0.6', linestyle=':', linewidth=1,
               label='95% energy')
    ax.axvline(0.5, color='0.5', linestyle=':', linewidth=1,
               label='Nyquist')

    if markers:
        for label, freq in markers.items():
            ax.axvline(freq, linestyle='--', linewidth=1.5, label=label)

    ax.set_xlabel('Frequency (cycles/voxel)')
    ax.set_ylabel('Cumulative Energy Fraction')
    ax.set_title(title or 'Cumulative Energy')
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
# Filter zoo — transfer functions for comparison
# ---------------------------------------------------------------------------

def _freq_grid_3d(shape):
    """Build 3D radial frequency magnitude grid for rFFT output."""
    rx, ry, rz = shape
    fx = fftfreq(rx)
    fy = fftfreq(ry)
    fz = rfftfreq(rz)
    FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
    freq_mag = np.sqrt(FX**2 + FY**2 + FZ**2)
    del FX, FY, FZ
    return freq_mag


def make_filter(name, cutoff, shape):
    """Create a 3D frequency-domain filter (transfer function).

    Parameters
    ----------
    name : str
        Filter type. One of:

        - ``'butterworth2'`` — Butterworth order 2 (current default)
        - ``'butterworth4'`` — Butterworth order 4 (sharper rolloff)
        - ``'butterworth8'`` — Butterworth order 8 (near brick-wall)
        - ``'gaussian'`` — Gaussian low-pass (sigma = cutoff)
        - ``'brick'`` — Ideal brick-wall (hard zero above cutoff)
        - ``'hann'`` — Hann (raised cosine) window, transition
          from cutoff*0.5 to cutoff*1.5
        - ``'tukey'`` — Tukey window (flat passband, cosine taper),
          transition from cutoff to cutoff*2

    cutoff : float
        Cutoff frequency in cycles/voxel.
    shape : tuple of int
        Volume shape (rx, ry, rz) — used to build frequency grid.

    Returns
    -------
    H : np.ndarray
        Transfer function, shape matching rfftn output.
    """
    freq_mag = _freq_grid_3d(shape)

    if name == 'butterworth2':
        H = 1.0 / (1.0 + (freq_mag / cutoff) ** 4)
    elif name == 'butterworth4':
        H = 1.0 / (1.0 + (freq_mag / cutoff) ** 8)
    elif name == 'butterworth8':
        H = 1.0 / (1.0 + (freq_mag / cutoff) ** 16)
    elif name == 'gaussian':
        H = np.exp(-0.5 * (freq_mag / cutoff) ** 2)
    elif name == 'brick':
        H = (freq_mag <= cutoff).astype(np.float32)
    elif name == 'hann':
        lo = cutoff * 0.5
        hi = cutoff * 1.5
        H = np.ones_like(freq_mag)
        trans = (freq_mag > lo) & (freq_mag <= hi)
        H[trans] = 0.5 * (1.0 + np.cos(
            np.pi * (freq_mag[trans] - lo) / (hi - lo)))
        H[freq_mag > hi] = 0.0
    elif name == 'tukey':
        lo = cutoff
        hi = cutoff * 2.0
        H = np.ones_like(freq_mag)
        trans = (freq_mag > lo) & (freq_mag <= hi)
        H[trans] = 0.5 * (1.0 + np.cos(
            np.pi * (freq_mag[trans] - lo) / (hi - lo)))
        H[freq_mag > hi] = 0.0
    else:
        raise ValueError(f"Unknown filter: {name!r}")

    return H.astype(np.float32)


def apply_filter(volume, name, cutoff):
    """Apply a named filter to a 3D volume in the frequency domain.

    Parameters
    ----------
    volume : np.ndarray
        3D float array.
    name : str
        Filter name (see ``make_filter``).
    cutoff : float
        Cutoff frequency in cycles/voxel.

    Returns
    -------
    np.ndarray
        Filtered volume (same shape as input).
    """
    from scipy.fft import irfftn
    H = make_filter(name, cutoff, volume.shape)
    F = rfftn(volume)
    F *= H
    return irfftn(F, s=volume.shape)


FILTER_ZOO = [
    ('butterworth2', 'Butterworth ord-2'),
    ('butterworth4', 'Butterworth ord-4'),
    ('butterworth8', 'Butterworth ord-8'),
    ('gaussian',     'Gaussian'),
    ('brick',        'Brick-wall (ideal)'),
    ('hann',         'Hann window'),
    ('tukey',        'Tukey window'),
]


def plot_filter_transfer_functions(cutoff=0.25, ax=None, title=None):
    """Plot 1D transfer function profiles for all filters in the zoo.

    Shows the gain vs frequency for each filter type at the given cutoff,
    evaluated along a single axis (1D cross-section through 3D filter).

    Parameters
    ----------
    cutoff : float
        Cutoff frequency.
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    f = np.linspace(0, 0.55, 500)

    for name, label in FILTER_ZOO:
        if name == 'butterworth2':
            H = 1.0 / (1.0 + (f / cutoff) ** 4)
        elif name == 'butterworth4':
            H = 1.0 / (1.0 + (f / cutoff) ** 8)
        elif name == 'butterworth8':
            H = 1.0 / (1.0 + (f / cutoff) ** 16)
        elif name == 'gaussian':
            H = np.exp(-0.5 * (f / cutoff) ** 2)
        elif name == 'brick':
            H = (f <= cutoff).astype(float)
        elif name == 'hann':
            lo, hi = cutoff * 0.5, cutoff * 1.5
            H = np.ones_like(f)
            trans = (f > lo) & (f <= hi)
            H[trans] = 0.5 * (1 + np.cos(np.pi * (f[trans] - lo) / (hi - lo)))
            H[f > hi] = 0.0
        elif name == 'tukey':
            lo, hi = cutoff, cutoff * 2.0
            H = np.ones_like(f)
            trans = (f > lo) & (f <= hi)
            H[trans] = 0.5 * (1 + np.cos(np.pi * (f[trans] - lo) / (hi - lo)))
            H[f > hi] = 0.0
        ax.plot(f, H, linewidth=1.5, label=label)

    ax.axvline(cutoff, color='k', linestyle=':', linewidth=1,
               label=f'cutoff ({cutoff})')
    ax.axvline(0.5, color='0.5', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Frequency (cycles/voxel)')
    ax.set_ylabel('Gain')
    ax.set_title(title or f'Filter Transfer Functions (cutoff={cutoff})')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.55)
    ax.set_ylim(-0.05, 1.1)
    return fig


def plot_filter_comparison(sdf, cutoff=0.25, filters=None,
                           n_bins=None, title=None):
    """Compare multiple filters on the same SDF: radial power spectra.

    Parameters
    ----------
    sdf : np.ndarray
        3D SDF (raw, before filtering).
    cutoff : float
        Cutoff frequency for all filters.
    filters : list of (name, label), optional
        Subset of FILTER_ZOO. Defaults to all.
    n_bins : int, optional
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if filters is None:
        filters = FILTER_ZOO

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top: overlay filtered spectra
    f_raw, p_raw = radial_power_spectrum(sdf, n_bins=n_bins)
    m = p_raw > 0
    axes[0].semilogy(f_raw[m], p_raw[m], 'k-', linewidth=2,
                     alpha=0.4, label='Raw SDF')
    for name, label in filters:
        filtered = apply_filter(sdf, name, cutoff)
        f_f, p_f = radial_power_spectrum(filtered, n_bins=n_bins)
        mf = p_f > 0
        axes[0].semilogy(f_f[mf], p_f[mf], linewidth=1.2, label=label)
        del filtered

    axes[0].axvline(cutoff, color='k', linestyle=':', linewidth=1)
    axes[0].axvline(0.5, color='0.5', linestyle=':', linewidth=0.8)
    axes[0].set_ylabel('Mean Power')
    axes[0].set_title(title or f'Filter Comparison (cutoff={cutoff})')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # Bottom: suppression ratio (raw / filtered) — how much each removes
    axes[1].axhline(1.0, color='k', linewidth=0.5)
    for name, label in filters:
        filtered = apply_filter(sdf, name, cutoff)
        f_f, p_f = radial_power_spectrum(filtered, n_bins=n_bins)
        ratio = np.where(p_f > 0, p_raw / (p_f + 1e-30), 1.0)
        axes[1].semilogy(f_f, ratio, linewidth=1.2, label=label)
        del filtered

    axes[1].axvline(cutoff, color='k', linestyle=':', linewidth=1)
    axes[1].axvline(0.5, color='0.5', linestyle=':', linewidth=0.8)
    axes[1].set_xlabel('Frequency (cycles/voxel)')
    axes[1].set_ylabel('Suppression Ratio (raw/filtered)')
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 0.55)
    fig.tight_layout()
    return fig
