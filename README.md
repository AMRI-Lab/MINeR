# MINeR

Official code for **"MINeR: Direction-modulated implicit neural representation enables ultrafast multi-shell diffusion MRI"**  
This repo holds the codes for MINeR.

---

## 1. Environmental Requirements

To run the reconstruction demo, the following dependencies are required:

- Python 3.10  
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

## 3. Usage
python main.py --mode train --demo_dir demo_data --output_dir outputs.
python main.py --mode infer --demo_dir demo_data --checkpoint outputs/model.pt --output_dir outputs.
