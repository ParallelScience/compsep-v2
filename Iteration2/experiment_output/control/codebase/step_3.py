# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os

def augment(data):
    aug_data = []
    for k in range(4):
        rotated = np.rot90(data, k=k, axes=(-2, -1))
        aug_data.append(rotated)
        aug_data.append(np.flip(rotated, axis=-2))
    return np.concatenate(aug_data, axis=0)

if __name__ == '__main__':
    print('Loading tSZ ground truth maps...')
    tsz_path = '/home/node/data/compsep_data/cut_maps/tsz.npy'
    tsz = np.load(tsz_path).astype(np.float32)
    print('Loading features...')
    features = np.load('data/features.npy').astype(np.float32)
    n_patches = tsz.shape[0]
    rng = np.random.default_rng(seed=42)
    indices = np.arange(n_patches)
    rng.shuffle(indices)
    n_train = int(n_patches * 0.70)
    n_val = int(n_patches * 0.15)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    print('Data split:')
    print('  Total patches: ' + str(n_patches))
    print('  Train patches: ' + str(len(train_idx)))
    print('  Val patches: ' + str(len(val_idx)))
    print('  Test patches: ' + str(len(test_idx)))
    train_tsz = tsz[train_idx]
    train_mean = np.mean(train_tsz)
    train_var = np.var(train_tsz)
    train_std = np.sqrt(train_var)
    print('\nNormalization statistics (computed on train set):')
    print('  tSZ Mean: ' + str(train_mean))
    print('  tSZ Variance: ' + str(train_var))
    print('  tSZ Std: ' + str(train_std))
    tsz_norm = (tsz - train_mean) / train_std
    masks = (tsz > 1e-7).astype(np.float32)
    print('\nSignificance masks (y > 10^-7):')
    print('  Overall mask positive fraction: ' + str(np.mean(masks)))
    print('  Train mask positive fraction: ' + str(np.mean(masks[train_idx])))
    print('  Val mask positive fraction: ' + str(np.mean(masks[val_idx])))
    print('  Test mask positive fraction: ' + str(np.mean(masks[test_idx])))
    train_features = features[train_idx]
    train_targets = tsz_norm[train_idx]
    train_masks = masks[train_idx]
    val_features = features[val_idx]
    val_targets = tsz_norm[val_idx]
    val_masks = masks[val_idx]
    test_features = features[test_idx]
    test_targets = tsz_norm[test_idx]
    test_masks = masks[test_idx]
    print('\nAugmenting training set (rotations and flips)...')
    train_features_aug = augment(train_features)
    train_targets_aug = augment(train_targets)
    train_masks_aug = augment(train_masks)
    print('  Original train features shape: ' + str(train_features.shape))
    print('  Augmented train features shape: ' + str(train_features_aug.shape))
    print('  Augmented train targets shape: ' + str(train_targets_aug.shape))
    print('  Augmented train masks shape: ' + str(train_masks_aug.shape))
    print('\nSaving processed datasets...')
    np.save('data/train_features.npy', train_features_aug)
    np.save('data/train_targets.npy', train_targets_aug)
    np.save('data/train_masks.npy', train_masks_aug)
    np.save('data/val_features.npy', val_features)
    np.save('data/val_targets.npy', val_targets)
    np.save('data/val_masks.npy', val_masks)
    np.save('data/test_features.npy', test_features)
    np.save('data/test_targets.npy', test_targets)
    np.save('data/test_masks.npy', test_masks)
    np.savez('data/tsz_norm_stats.npz', mean=train_mean, var=train_var, std=train_std)
    np.savez('data/split_indices.npz', train=train_idx, val=val_idx, test=test_idx)
    print('Saved all files to data/ directory.')