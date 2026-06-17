from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

import argparse
import nibabel as nib
import numpy as np
import torch

import utils

from miner_core import build_miner_model
from miner_core import compute_regularization_loss
from miner_core import prepare_qspace


def load_bvecs(path):
    bvecs = np.loadtxt(path).astype(np.float32)

    if bvecs.ndim != 2:
        raise ValueError(f"Expected a 2D b-vector file, got shape {bvecs.shape}")

    if bvecs.shape[0] == 3:
        bvecs = bvecs.T

    if bvecs.shape[1] != 3:
        raise ValueError(
            f"Expected b-vectors with shape (3, N) or (N, 3), got {bvecs.shape}"
        )

    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvecs = bvecs / np.clip(norms, 1e-8, None)

    return bvecs


def load_mask(path, image_shape):
    mask = nib.load(str(path)).get_fdata().astype(np.float32)

    if mask.ndim == 2:
        mask = mask[..., None]
    elif mask.ndim == 3:
        if mask.shape[-1] == 1:
            pass
        elif mask.shape[:2] == image_shape:
            mask = mask[..., :1]
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    if mask.shape[:2] != image_shape:
        raise ValueError(
            f"Mask shape {mask.shape[:2]} does not match image shape {image_shape}"
        )

    mask = (mask > 0).astype(np.float32)

    return mask


def load_demo_data(demo_dir, device):
    demo_dir = Path(demo_dir)
    if not demo_dir.is_absolute():
        demo_dir = PROJECT_ROOT / demo_dir
    dwi_path = demo_dir / "dwi.nii"
    bvecs_path = demo_dir / "bvecs.txt"
    bvecs_all_path = demo_dir / "bvecs_all.txt"
    mask_path = demo_dir / "mask.nii.gz"

    for path in [dwi_path, bvecs_path, bvecs_all_path, mask_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing demo file: {path}")

    dwi_img = nib.load(str(dwi_path))
    dwi = dwi_img.get_fdata().astype(np.float32)

    if dwi.ndim != 3:
        raise ValueError(f"Expected dwi.nii with shape (H, W, C), got {dwi.shape}")

    h, w, c = dwi.shape

    bvecs = load_bvecs(bvecs_path)
    bvecs_all = load_bvecs(bvecs_all_path)

    if bvecs.shape[0] != c:
        raise ValueError(
            f"dwi.nii has {c} channels, but bvecs.txt has {bvecs.shape[0]} directions"
        )

    if bvecs_all.shape[0] <= bvecs.shape[0]:
        raise ValueError(
            f"bvecs_all.txt should contain more target directions than bvecs.txt. "
            f"Got {bvecs_all.shape[0]} and {bvecs.shape[0]}."
        )

    mask = load_mask(mask_path, image_shape=(h, w))
    valid = mask.squeeze(-1) > 0

    if not np.any(valid):
        raise ValueError("The mask is empty.")

    dwi_min = dwi[valid].min()
    dwi_max = dwi[valid].max()
    dwi = (dwi - dwi_min) / (dwi_max - dwi_min + 1e-8)
    dwi = np.clip(dwi, 0.0, 1.0)

    coords = torch.from_numpy(utils.generate_coords(h, w)).float().to(device)
    dwi = torch.from_numpy(dwi).float().to(device)
    mask = torch.from_numpy(mask).float().to(device)

    directions_train = torch.from_numpy(bvecs).float().to(device)
    directions_all = torch.from_numpy(bvecs_all).float().to(device)

    qspace = prepare_qspace(
        train_directions=directions_train,
        full_directions=directions_all,
    )

    return {
        "coords": coords,
        "dwi_train": dwi,
        "mask": mask,
        "qspace": qspace,
        "affine": dwi_img.affine,
        "image_shape": (h, w),
        "dwi_min": float(dwi_min),
        "dwi_max": float(dwi_max),
    }


def make_configs():
    encoding_config = {
        "otype": "HashGrid",
        "n_levels": 24,
        "n_features_per_level": 4,
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 1.8,
    }

    network_config = {
        "num_layers": 5,
        "hidden_dim": 512,
        "activation": "LeakyReLU",
    }

    return encoding_config, network_config



def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_demo_data(args.demo_dir, device)
    encoding_config, network_config = make_configs()

    model = build_miner_model(
        encoding_config=encoding_config,
        network_config=network_config,
        image_shape=data["image_shape"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    mse_loss = torch.nn.MSELoss()



    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        pred_train = model(data["coords"], data["qspace"].train)

        loss_sup = mse_loss(
            pred_train * data["mask"],
            data["dwi_train"] * data["mask"],
        )

        if epoch >= args.warmup:
            pred_all = model(data["coords"], data["qspace"].full)

            loss_reg = compute_regularization_loss(
                pred=pred_all,
                qspace=data["qspace"],
                mask=data["mask"],
            )

            loss = loss_sup + args.reg_weight * loss_reg
        else:
        
            loss = loss_sup

        loss.backward()
        optimizer.step()



    ckpt_path = output_dir / "model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_shape": data["image_shape"],
            "dwi_min": data["dwi_min"],
            "dwi_max": data["dwi_max"],
        },
        ckpt_path,
    )

    print(f"Training finished. Checkpoint saved to: {ckpt_path}")


def save_prediction(model, coords, directions, mask, affine, path):
    model.eval()

    with torch.no_grad():
        pred = model(coords, directions)
        pred = pred * mask

    pred_np = pred.detach().cpu().numpy()
    nib.save(nib.Nifti1Image(pred_np, affine), str(path))

    print(f"Prediction saved to: {path}")

def infer(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    data = load_demo_data(args.demo_dir, device)
    encoding_config, network_config = make_configs()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_miner_model(
        encoding_config=encoding_config,
        network_config=network_config,
        image_shape=tuple(checkpoint["image_shape"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    out_path = output_dir / "pred_dwi.nii.gz"

    save_prediction(
        model=model,
        coords=data["coords"],
        directions=data["qspace"].full,
        mask=data["mask"],
        affine=data["affine"],
        path=out_path,
    )

    print(f"Inference finished. Result saved to: {out_path}")

def parse_args():
    parser = argparse.ArgumentParser()

    #parser.add_argument("--mode", type=str, choices=["train", "infer"], required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "infer"],
        default="infer"
    )
    parser.add_argument("--demo_dir", type=str, default="demo_data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint", type=str, default="outputs/model.pt")

    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--reg_weight", type=float, default=0.05)

    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
    else:
        infer(args)
