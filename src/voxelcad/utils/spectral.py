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
        n_bins = min(rx, ry, rz) // 2

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
