# From Movement Primitives to Distance Fields to Dynamical Systems

[📄 Paper](https://arxiv.org/pdf/2504.09705) | [🌐 Interactive Webpage](https://mp-df-ds.github.io/)

---
**A simple module to represent trajectories using quadratic splines, enabling smooth transitions from movement primitives to distance fields and dynamical systems—all with analytical gradients and PyTorch support.**

---

## ✨ What is this?

This project provides a simple and lightweight implementation to convert a trajectory into:

- **Movement Primitives (MP)**
- **Distance Fields (DF)**
- **Dynamical Systems (DS)**

by representing it as a series of **concatenated quadratic splines**. Thanks to the analytical gradients, it's easy to compute distances and directions at any point around the trajectory.

---

## 🚀 Key Features

- ✅ Minimal dependencies (built with **PyTorch**, no heavy libraries needed)
- ✅ Fully vectorized and **parallelizable**
- ✅ Supports **gradient-based learning**, **optimization**, and **control**
- ✅ **Efficient** computation



## 📂 Project Structure

| File | Description |
|------|-------------|
| `data` | Trajectories for testing|
| `quadratic_spline.py` | Core implementation of spline representation and gradient computation |
| `run_mp_df_ds.py` | Example: Convert a quadratic spline into distance field, and dynamical system |
| `run_single_traj.py` | Similar to above, but for a trajectory that represented using discrete points|
| `run_multiple_traj.py` | Combine and fuse multiple trajectories |
| `run_LASA.py` | Run experiments on the LASA dataset (requires [pylasadataset](https://github.com/justagist/pyLasaDataset)) 
---



