import argparse
import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

import MINeR
def evaluate_interpolation(model, coords, theta, phi, mask, dwi_shape, affine, result_path, epoch, device):
    model.eval()
    with torch.no_grad():
        pred, _ = model(coords, theta, phi)
        interpolated_dwi = pred.view(dwi_shape[0], dwi_shape[1], -1)
        interpolated_dwi = interpolated_dwi * mask
        interpolated = interpolated_dwi.cpu().detach().numpy()
        output_filename = os.path.join(result_path, f'interpolated_dwi_epoch_{epoch}.nii.gz')
        nib.save(nib.Nifti1Image(interpolated, affine), output_filename)
        print(f"Interpolated DWI saved at {output_filename}")
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Run MINeR demo")

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--summary_epoch', type=int, default=200)
    parser.add_argument('--data_dir', type=str, default='demo_data')
    parser.add_argument('--output_dir', type=str, default='results_demo')

    parser.add_argument('--n_neurons', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=9)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--rank', type=int, default=10)
    parser.add_argument('--n_levels', type=int, default=24)
    parser.add_argument('--n_features_per_level', type=int, default=4)
    parser.add_argument('--log2_hashmap_size', type=int, default=19)
    parser.add_argument('--base_resolution', type=int, default=16)
    parser.add_argument('--per_level_scale', type=float, default=1.8)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dwi_path = os.path.join(args.data_dir, "dwi_demo.nii")
    mask_path = os.path.join(args.data_dir, "mask.nii")
    train_bvecs_path = os.path.join(args.data_dir, "bvecs_demo.txt")
    interp_bvecs_path = os.path.join(args.data_dir, "bvecs_interp_demo.txt")

    dwi_img = nib.load(dwi_path)
    mask = nib.load(mask_path).get_fdata()
    dwi = dwi_img.get_fdata()
    affine = dwi_img.affine
    bvecs = np.loadtxt(train_bvecs_path)
    interp_bvecs = np.loadtxt(interp_bvecs_path)
    H, W, C = dwi.shape

    encoding_config = {
        "otype": "HashGrid",
        "n_levels": args.n_levels,
        "n_features_per_level": args.n_features_per_level,
        "log2_hashmap_size": args.log2_hashmap_size,
        "base_resolution": args.base_resolution,
        "per_level_scale": args.per_level_scale,
    }

    model = MINeR.build_model(
        n_neurons=args.n_neurons,
        n_layers=args.n_layers,
        num_dir=C,
        device=device,
        latent_dim=args.latent_dim,
        rank=args.rank,
        encoding_config=encoding_config
    )
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    dwi_min, dwi_max = dwi.min(), dwi.max()
    dwi_norm = (dwi - dwi_min) / (dwi_max - dwi_min)
    coords = torch.from_numpy(MINeR.generate_coords(H, W)).to(device).float()
    dwi_t = torch.from_numpy(dwi_norm).to(device).float()
    mask_t = torch.from_numpy(mask).to(device).float().unsqueeze(-1)
    theta = torch.from_numpy(np.arccos(bvecs[2])).to(device).float()
    phi = torch.from_numpy(np.arctan2(bvecs[1], bvecs[0])).to(device).float()
    theta_interp = torch.from_numpy(np.arccos(interp_bvecs[2])).to(device).float()
    phi_interp = torch.from_numpy(np.arctan2(interp_bvecs[1], interp_bvecs[0])).to(device).float()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, 'model.pkl')
    log_file_path = os.path.join(args.output_dir, 'training_log.txt')

    # checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        log_file = open(log_file_path, 'a')
        print(f"Loaded checkpoint at epoch {start_epoch}")
    else:
        log_file = open(log_file_path, 'w')
        start_epoch = 0
        print("No checkpoint found. Training from scratch.")

    for e in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(coords, theta, phi)
        loss = criterion(pred * mask_t.view(-1, C), dwi_t.view(-1, C) * mask_t.view(-1, C))
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        print(f'Epoch [{e+1}/{args.epochs}], Loss: {loss.item():.8f}')
        log_file.write(f'Epoch [{e+1}/{args.epochs}], Loss: {loss.item():.8f}\n')

        if (e + 1) % args.summary_epoch == 0:
            evaluate_interpolation(model, coords, theta_interp, phi_interp,
                                   mask_t, (H, W), affine, args.output_dir, e + 1, device)

        checkpoint = {
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, checkpoint_path)

    log_file.close()



if __name__ == "__main__":
    main()
