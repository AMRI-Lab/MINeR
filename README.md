# MINeR

Official code for **"MINeR: Direction-modulated implicit neural representation enables ultrafast multi-shell diffusion MRI"**  
This repo holds the codes for MINeR.

---

## 1. Environmental Requirements

To run the reconstruction demo, the following dependencies are required:

- Python 3.10  
- PyTorch 2.1.0  
- tinycudann 1.7  
- numpy 1.23.5  
- nibabel 5.1.0  
- (optional) matplotlib, tensorboard, h5py

---

## 2. Sample Data

This repo provides a **toy undersampled demo dataset** in `demo_data/`.  
Full datasets can be downloaded from the **HCP (Human Connectome Project)** database.

---

## 3. Files Descriptions
```
MINeR/
├── run_demo.py # Demo script for training & interpolation
├── MINeR.pyc # Compiled core implementation
└── demo_data/ # Toy undersampled DWI data
```
---

## 4. Usage

To test the performance of MINeR, run the following command:

```bash
python run_demo.py --gpu 0 --epochs 1000 --summary_epoch 200
