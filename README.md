# MINeR
Official code for **"MINeR: Direction-modulated implicit neural representation enables ultrafast multi-shell diffusion MRI"**  
This repo holds the codes for MINeR.

Direction-modulated hash INR for DWI interpolation (6 -> 30 directions).
A hash grid encodes `(x, y)`; a direction latent drives a hypernetwork that
generates the first-layer weight of the base network,
`W1 = L2norm(tanh(U V))` (paper Eq. 5), and the remaining layers are shared
across directions. Training uses masked MSE plus a geometry-guided
interpolation loss in the log-signal domain (Eqs. 6-9).


---

## 1. Environmental Requirements

To run the reconstruction demo, the following dependencies are required:

- Python 3.8
- PyTorch 2.0.0  
- tinycudann 1.7  
- numpy 1.22.4  
- nibabel 5.3.2  
- (optional) matplotlib, tensorboard, h5py

---

## 2. Sample Data

This repo provides a **toy undersampled demo dataset** in `demo_data/`.  
Full datasets can be downloaded from the **HCP (Human Connectome Project)** database.

---

## 3. Quick start
```bash
cd MINeR
python main.py --mode train                              # 800 epochs
python main.py --mode infer --ckpt runs/demo/model.pkl   # 30-direction output
python compare_metrics.py --gt demo_data/dwi_all.nii.gz  # metrics + figures
```
