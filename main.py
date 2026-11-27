# -*- coding: utf-8 -*-
"""MINeR v2 -- training and inference entry point.

Everything model-related (hash encoder, direction-modulated INR, losses,
metrics) lives in ``miner_core``; this file only drives the data flow.

Training (6 -> 30 directions, paper-aligned configuration):

    python main.py --mode train

Inference with a trained checkpoint:

    python main.py --mode infer --ckpt runs/demo/model.pkl

The demo data in ``demo_data`` is slice 69 of the b=1000 shell:
  dwi.nii.gz    6 measured DWIs    [145, 145, 6]
  bvecs.txt     6 direction vectors (one per line)
  bvecs_all.txt 30 direction vectors (one per line, order of the prediction)
  mask.nii.gz   brain mask         [145, 145, 1]

Metrics are computed on the full image.
"""
# Sampling condition: each interpolated direction m must lie on the great-circle
# arc between two sampled directions a and b, i.e. alpha = theta(a,m)/theta(a,b)
# must be in [0, 1] with a small out-of-plane deviation. For the 6 -> 30 setup
# used here, the interpolated directions are 12-41 deg from the nearest sampled
# direction and the sampled pairs span 43-77 deg; beyond roughly 40-45 deg to the
# nearest sample (or ~80 deg between samples) the log-linear interpolation
# assumption and the reconstruction accuracy degrade, so more sampled directions
# or an angular physical model are required.

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from miner_core import (DEFAULT_TRIPLETS, DirectionModulatedHashINR,
                        generate_coords, interpolation_loss, load_bvecs,
                        load_mask, masked_mse, minmax_normalize, save_nifti,
                        set_seed, triplet_report)


def parse_known(text):
    return [int(v) for v in text.split(",") if v.strip() != ""]


def build_model(args):
    encoder_cfg = {"otype": "HashGrid", "n_levels": 24,
                   "n_features_per_level": 4, "log2_hashmap_size": 19,
                   "base_resolution": 16, "per_level_scale": 1.8}
    network_cfg = {"num_layers": args.layers, "hidden_dim": 512,
                   "activation": "LeakyReLU"}
    return DirectionModulatedHashINR(
        encoding_config=encoder_cfg, network_config=network_cfg,
        n_input_dims=2, direction_dim=3, latent_dim=args.latent_dim, rank=args.rank,
        modulation="paper", grad_checkpoint=args.grad_checkpoint,
        encoder_backend=args.encoder, encoder_seed=args.seed)


def load_data(data_dir, known):
    dwi_path = os.path.join(data_dir, "dwi.nii.gz")
    img = nib.load(dwi_path)
    dwi = np.asarray(img.get_fdata(dtype=np.float32))      # [H, W, 6]
    dwi = np.squeeze(dwi)                                  # tolerate [H, W, 1, 6]
    affine = img.affine
    if dwi.ndim != 3:
        raise ValueError("dwi.nii.gz must be [H, W, n_supervised], got %s"
                         % (dwi.shape,))
    mask = load_mask(os.path.join(data_dir, "mask.nii.gz"))  # [H, W]
    bvecs = load_bvecs(os.path.join(data_dir, "bvecs.txt"))  # [n_sup, 3]
    bvecs_all = load_bvecs(os.path.join(data_dir, "bvecs_all.txt"))
    if dwi.shape[-1] != len(bvecs):
        raise ValueError("dwi has %d directions but bvecs.txt has %d"
                         % (dwi.shape[-1], len(bvecs)))
    if len(bvecs_all) <= len(bvecs):
        raise ValueError("bvecs_all.txt must contain the interpolated directions "
                         "as well")
    if known and max(known) >= len(bvecs_all):
        raise ValueError("supervised index %d out of range for %d directions"
                         % (max(known), len(bvecs_all)))
    if known:
        diff = np.abs(bvecs_all[known] - bvecs).max()
        if diff > 1e-3:
            print("WARNING: bvecs.txt does not match bvecs_all.txt[%s] "
                  "(max deviation %.3g)" % (known, diff))
    return dwi, affine, mask, bvecs, bvecs_all


def to_image(flat, h, w, n):
    return flat.reshape(h, w, n)


def predict_all(model, coords, g_all, h, w):
    model.eval()
    with torch.no_grad():
        return to_image(model(coords, g_all), h, w, g_all.shape[0])


def save_snapshot(model, coords, g_known, g_all, mask, affine, out_dir, epoch,
                  prefix="interp_epoch"):
    h, w = mask.shape
    pred = predict_all(model, coords, g_all, h, w).cpu().numpy()
    save_nifti((pred * mask[..., None])[:, :, None, :], affine,
               os.path.join(out_dir, "%s_%d.nii.gz" % (prefix, epoch + 1)))
    pred_known = predict_all(model, coords, g_known, h, w).cpu().numpy()
    save_nifti((pred_known * mask[..., None])[:, :, None, :], affine,
               os.path.join(out_dir, "pre_epoch_%d.nii.gz" % (epoch + 1)))


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    known = parse_known(args.known)
    dwi, affine, mask, bvecs, bvecs_all = load_data(args.data_dir, known)
    h, w = mask.shape
    dwi_norm, lo, hi = minmax_normalize(dwi)
    g_known = torch.from_numpy(bvecs).to(device).float()
    g_all = torch.from_numpy(bvecs_all).to(device).float()
    coords = torch.from_numpy(generate_coords(h, w)).to(device).float()
    target = torch.from_numpy(dwi_norm).to(device).float()
    mask_t = torch.from_numpy(mask).to(device).float()

    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=args.scheduler_patience,
            factor=0.5, min_lr=args.min_lr)

    print("MINeR v2 | device=%s | %d -> %d directions | %d epochs"
          % (device, len(known), len(bvecs_all), args.epochs))
    print("supervised indices: %s | lam=%.3g | warm-up=%d"
          % (known, args.lam, args.warm))
    print("alpha of the %d triplets: %s" %
          (len(DEFAULT_TRIPLETS),
           ", ".join("%.3f" % a for _, _, _, a in triplet_report(g_all))))
    n_par = sum(p.numel() for p in model.parameters())
    print("parameters: %.2f M (encoder %.2f M)"
          % (n_par / 1e6, sum(p.numel() for p in model.encoding.parameters()) / 1e6))

    log_path = os.path.join(out_dir, "training_log.txt")
    log_file = open(log_path, "w")
    history, interp_hist = [], []
    t0 = time.time()

    for e in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        pred_known = to_image(model(coords, g_known), h, w, len(known))
        loss_sup = masked_mse(pred_known, target, mask_t)
        if e >= args.warm:
            pred_all = to_image(model(coords, g_all), h, w, len(bvecs_all))
            loss_interp = interpolation_loss(pred_all, g_all, mask=mask_t)
            loss = loss_sup + args.lam * loss_interp
        else:
            loss_interp = loss_sup.new_zeros(())
            loss = loss_sup
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss.item())
        history.append(float(loss))
        interp_hist.append(float(loss_interp))
        log_file.write("Epoch [%d/%d] loss:%.6f loss_sup:%.6f loss_interp:%.6f\n"
                       % (e + 1, args.epochs, loss.item(), loss_sup.item(),
                          loss_interp.item()))
        if (e + 1) % args.print_every == 0 or e == 0 or e + 1 == args.epochs:
            print("Epoch [%d/%d] loss:%.6f sup:%.6f interp:%.6f  (%.0fs)"
                  % (e + 1, args.epochs, loss.item(), loss_sup.item(),
                     loss_interp.item(), time.time() - t0))
        if (e + 1) % args.save_every == 0 or e + 1 == args.epochs:
            save_snapshot(model, coords, g_known, g_all, mask, affine, out_dir, e)
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": e, "config": vars(args)},
                       os.path.join(out_dir, "model_last.pkl"))

    log_file.close()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs - 1,
        "config": vars(args),
        "normalization": {"lo": lo, "hi": hi},
        "supervised_indices": known,
    }
    torch.save(checkpoint, os.path.join(out_dir, "model.pkl"))
    print("checkpoint -> %s" % os.path.join(out_dir, "model.pkl"))

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(history) + 1), history, label="total")
    plt.plot(range(1, len(interp_hist) + 1), interp_hist, label="interpolation")
    plt.axvline(args.warm, color="gray", linestyle="--",
                label="interp loss on (epoch %d)" % args.warm)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=110)
    plt.close()

def infer(args):
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    if not args.ckpt:
        raise SystemExit("--ckpt is required for --mode infer")
    ckpt = torch.load(args.ckpt, map_location=device)
    known = ckpt.get("supervised_indices", parse_known(args.known))
    dwi, affine, mask, bvecs, bvecs_all = load_data(args.data_dir, known)
    h, w = mask.shape
    model = build_model(args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    coords = torch.from_numpy(generate_coords(h, w)).to(device).float()
    g_all = torch.from_numpy(bvecs_all).to(device).float()
    pred = predict_all(model, coords, g_all, h, w).cpu().numpy()
    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "interp.nii.gz")
    save_nifti((pred * mask[..., None])[:, :, None, :], affine, out_path)
    print("predicted %d directions -> %s" % (len(bvecs_all), out_path))


def main():
    ap = argparse.ArgumentParser(description="MINeR v2 (6 -> 30 direction INR)")
    ap.add_argument("--mode", choices=["train", "infer"], default="train")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "demo_data"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "runs", "demo"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--warm", type=int, default=300,
                    help="interpolation loss is switched on at this epoch")
    ap.add_argument("--lam", type=float, default=0.05,
                    help="weight of the interpolation loss")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--scheduler", action="store_true",
                    help="enable ReduceLROnPlateau")
    ap.add_argument("--scheduler-patience", type=int, default=10)
    ap.add_argument("--min-lr", type=float, default=1e-5,
                    help="lower bound of the scheduler")
    ap.add_argument("--layers", type=int, default=9)
    ap.add_argument("--rank", type=int, default=10)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--known", default="0,6,12,18,24,29",
                    help="indices of the supervised directions in bvecs_all.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--encoder", choices=["pytorch", "tcnn", "auto"],
                    default="pytorch")
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint",
                    action="store_false")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--print-every", type=int, default=20)
    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
