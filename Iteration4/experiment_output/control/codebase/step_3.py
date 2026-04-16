# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
mpl.rcParams['text.usetex'] = False
sys.path.insert(0, os.path.abspath('codebase'))
sys.path.insert(0, '/home/node/data/compsep_data/')
import utils
from step_1 import CompSepDataset
from step_2 import UNet
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    with open('data/splits.json', 'r') as f:
        splits = json.load(f)
    val_idx = splits['val']
    print('Validation samples: ' + str(len(val_idx)))
    val_dataset = CompSepDataset(val_idx, split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    model = UNet(in_channels=6, out_channels=1, cond_dim=6, features=[64, 128, 256, 512]).to(device)
    model.load_state_dict(torch.load('data/sr_dae_model.pth', map_location=device))
    model.eval()
    transfer_functions = []
    ells = None
    print('Computing transfer functions...')
    with torch.no_grad():
        for x, noise_vars, y in val_loader:
            x = x.to(device)
            noise_vars = torch.log10(noise_vars + 1e-8).to(device)
            pred = model(x, noise_vars)
            pred_np = pred.cpu().numpy() / 1e6
            y_np = y.cpu().numpy()
            for i in range(pred_np.shape[0]):
                p_np = pred_np[i, 0]
                t_np = y_np[i]
                out1, out2 = utils.powers(p_np, p_np, ps=1.17, window_alpha=0.5)
                if np.max(out1) > np.max(out2):
                    ell = out1
                    ps_pred = out2
                else:
                    ell = out2
                    ps_pred = out1
                out1_t, out2_t = utils.powers(t_np, t_np, ps=1.17, window_alpha=0.5)
                if np.max(out1_t) > np.max(out2_t):
                    ps_true = out2_t
                else:
                    ps_true = out1_t
                if ells is None:
                    ells = np.array(ell)
                t_ell = np.sqrt(np.abs(ps_pred) / (np.abs(ps_true) + 1e-20))
                transfer_functions.append(t_ell)
    transfer_functions = np.array(transfer_functions)
    mean_t_ell = np.mean(transfer_functions, axis=0)
    std_t_ell = np.std(transfer_functions, axis=0)
    mask = (ells > 3000) & (ells < 8000)
    if np.any(mask):
        mean_t_ell_range = np.mean(mean_t_ell[mask])
    else:
        mean_t_ell_range = np.nan
    print('Mean T(ell) in 3000 < ell < 8000: ' + str(mean_t_ell_range))
    np.savez('data/transfer_function.npz', ell=ells, mean_t_ell=mean_t_ell, std_t_ell=std_t_ell, mean_t_ell_3000_8000=mean_t_ell_range)
    print('Transfer function and metrics saved to data/transfer_function.npz')
    plt.figure(figsize=(10, 6))
    mask_valid = ells > 0
    ells_plot = ells[mask_valid]
    mean_t_ell_plot = mean_t_ell[mask_valid]
    std_t_ell_plot = std_t_ell[mask_valid]
    plt.plot(ells_plot, mean_t_ell_plot, label='Mean T(ell)', color='blue')
    plt.fill_between(ells_plot, mean_t_ell_plot - std_t_ell_plot, mean_t_ell_plot + std_t_ell_plot, color='blue', alpha=0.3, label='+/- 1 std')
    if np.max(ells_plot) > 5000:
        plt.axvspan(5000, np.max(ells_plot), color='red', alpha=0.1, label='ell > 5000 regime')
    plt.axhline(1.0, color='k', linestyle='--', label='Ideal')
    plt.xlabel('Multipole ell')
    plt.ylabel('Transfer Function T(ell)')
    plt.title('Transfer Function of SR-DAE Denoising')
    plt.xscale('log')
    plt.xlim(np.min(ells_plot), np.max(ells_plot))
    plt.ylim(0, 2)
    plt.legend()
    plt.grid(True, which='both', ls='-', alpha=0.2)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = 'data/transfer_function_1_' + timestamp + '.png'
    plt.savefig(plot_filename, dpi=300)
    print('Plot saved to ' + plot_filename)
if __name__ == '__main__':
    main()