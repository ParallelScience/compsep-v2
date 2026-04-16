# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib

matplotlib.rcParams['text.usetex'] = False

sys.path.insert(0, '/home/node/data/compsep_data')
import utils

def verify_powers_estimator():
    np.random.seed(42)
    n_pixels = 256
    patch_size_deg = 5.0
    n_realizations = 20
    pix_size_rad = (patch_size_deg * np.pi / 180.0) / n_pixels
    omega_pix = pix_size_rad**2
    theoretical_cl = omega_pix
    cl_avg = None
    ell = None
    for _ in range(n_realizations):
        noise_map = np.random.randn(n_pixels, n_pixels)
        res = utils.powers(noise_map, noise_map, ps=patch_size_deg)
        if np.mean(res[0]) > np.mean(res[1]):
            ell_curr, cl_curr = res[0], res[1]
        else:
            ell_curr, cl_curr = res[1], res[0]
        if cl_avg is None:
            cl_avg = cl_curr
            ell = ell_curr
        else:
            cl_avg += cl_curr
    cl_avg /= n_realizations
    valid = ell > 0
    ell = ell[valid]
    cl_avg = cl_avg[valid]
    return ell, cl_avg, theoretical_cl

def check_coordinate_alignment(base_dir):
    tsz_maps = np.load(os.path.join(base_dir, 'tsz.npy'))
    cross_corr_avg = np.zeros((256, 256))
    auto_corr_avg = 0.0
    for i in range(len(tsz_maps)):
        tsz_patch = tsz_maps[i]
        tsz_rot = np.rot90(tsz_patch)
        tsz_patch_centered = tsz_patch - np.mean(tsz_patch)
        tsz_rot_centered = tsz_rot - np.mean(tsz_rot)
        corr = signal.correlate(tsz_patch_centered, tsz_rot_centered, mode='same', method='fft')
        cross_corr_avg += corr
        auto_corr_avg += np.sum(tsz_patch_centered**2)
    if auto_corr_avg > 0:
        cross_corr_avg /= auto_corr_avg
    return cross_corr_avg

def compute_numerical_diagnostics(base_dir, components, units):
    stats = {}
    print("Numerical Diagnostics for Ground-Truth Component Maps:")
    print("-" * 105)
    header = "Component  | Units      | Mean        | Variance    | Min         | Max         | 1st Pctl    | 99th Pctl  "
    print(header)
    print("-" * 105)
    for name, filename in components.items():
        filepath = os.path.join(base_dir, filename)
        data = np.load(filepath)
        mean_val = np.mean(data)
        var_val = np.var(data)
        min_val = np.min(data)
        max_val = np.max(data)
        p01 = np.percentile(data, 1)
        p99 = np.percentile(data, 99)
        stats[name] = {'mean': mean_val, 'var': var_val, 'min': min_val, 'max': max_val, 'p01': p01, 'p99': p99}
        unit = units[name]
        line = name.ljust(10) + " | " + unit.ljust(10) + " | " + ("%.3e" % mean_val).rjust(11) + " | " + ("%.3e" % var_val).rjust(11) + " | " + ("%.3e" % min_val).rjust(11) + " | " + ("%.3e" % max_val).rjust(11) + " | " + ("%.3e" % p01).rjust(11) + " | " + ("%.3e" % p99).rjust(11)
        print(line)
        del data
    print("-" * 105)
    return stats

def generate_diagnostic_plot(ell, cl_avg, theoretical_cl, cross_corr_map, stats):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].plot(ell, cl_avg, label='Computed C_ell (Avg of 20)', color='blue')
    axes[0].axhline(theoretical_cl, color='red', linestyle='--', label='Theoretical C_ell')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Multipole ell')
    axes[0].set_ylabel('Power Spectrum C_ell')
    axes[0].set_title('White Noise Power Spectrum Check')
    axes[0].legend()
    axes[0].grid(True, which='both', linestyle=':', alpha=0.6)
    im = axes[1].imshow(cross_corr_map, cmap='viridis', origin='lower', extent=[-128, 128, -128, 128], vmin=-1, vmax=1)
    axes[1].set_title('2D Cross-Correlation (Avg over all patches):\ntSZ vs 90-deg Rotated tSZ\n(Expected: ~0 everywhere)')
    axes[1].set_xlabel('X shift (pixels)')
    axes[1].set_ylabel('Y shift (pixels)')
    axes[1].axhline(0, color='white', linestyle='--', alpha=0.5)
    axes[1].axvline(0, color='white', linestyle='--', alpha=0.5)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Correlation Coefficient')
    max_corr = np.max(np.abs(cross_corr_map))
    axes[1].text(0.05, 0.95, "Max |corr| = " + ("%.4f" % max_corr), transform=axes[1].transAxes, color='white', fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    names = list(stats.keys())
    dynamic_ranges = [stats[n]['p99'] - stats[n]['p01'] for n in names]
    axes[2].bar(names, dynamic_ranges, color='coral', edgecolor='black')
    axes[2].set_yscale('log')
    axes[2].set_ylabel('Dynamic Range (99th - 1st Pctl) [Native Units]')
    axes[2].set_title('Dynamic Range of Components')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, axis='y', linestyle=':', alpha=0.6)
    unit_text = "Units:\nCMB: uK_CMB\ntSZ, kSZ: dimensionless\nCIB: Jy/sr"
    axes[2].text(0.95, 0.05, unit_text, transform=axes[2].transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout(pad=2.0)
    timestamp = str(int(time.time()))
    plot_filename = os.path.join("data", "diagnostic_plot_1_" + timestamp + ".png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print("Diagnostic plot saved to " + plot_filename)

if __name__ == '__main__':
    print("Starting Evaluation Pipeline and Coordinate Validation...")
    base_directory = '/home/node/data/compsep_data/cut_maps'
    print("Verifying utils.powers estimator with Gaussian white noise...")
    ell_vals, cl_vals, cl_theory = verify_powers_estimator()
    print("Performing coordinate alignment check...")
    cross_correlation_map = check_coordinate_alignment(base_dir=base_directory)
    print("Computing numerical diagnostics for ground-truth component maps...")
    component_files = {'CMB': 'lensed_cmb.npy', 'tSZ': 'tsz.npy', 'kSZ': 'ksz.npy', 'CIB 90': 'cib_90.npy', 'CIB 150': 'cib_150.npy', 'CIB 217': 'cib_217.npy', 'CIB 353': 'cib_353.npy', 'CIB 545': 'cib_545.npy', 'CIB 857': 'cib_857.npy'}
    component_units = {'CMB': 'uK_CMB', 'tSZ': 'dim.less', 'kSZ': 'dim.less', 'CIB 90': 'Jy/sr', 'CIB 150': 'Jy/sr', 'CIB 217': 'Jy/sr', 'CIB 353': 'Jy/sr', 'CIB 545': 'Jy/sr', 'CIB 857': 'Jy/sr'}
    statistics = compute_numerical_diagnostics(base_dir=base_directory, components=component_files, units=component_units)
    print("Generating multi-panel diagnostic plot...")
    generate_diagnostic_plot(ell=ell_vals, cl_avg=cl_vals, theoretical_cl=cl_theory, cross_corr_map=cross_correlation_map, stats=statistics)