# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yiming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


import torch
import matplotlib.pyplot as plt
from quadratic_spline import QuadraticSpline,dynamical_system_single_step
import numpy as np

if __name__ == "__main__":
    device = 'cuda'
    curve = QuadraticSpline(nbFct=3, nbSeg=10,device=device)

    # Create a trajectory concatenating two letters
    data_dir = ""
    CShape = np.load(data_dir + "S.npy", allow_pickle="True")
    CShape = torch.from_numpy(CShape).to(device)
    pos,vel,acc = CShape[0,:,:2],CShape[0,:,2:4],CShape[0,:,4:]
    # Encode trajectory
    x_min,x_max = pos[:,0].min()-5.0,pos[:,0].max()+5.0
    y_min,y_max = pos[:,1].min()-5.0,pos[:,1].max()+5.0

    w_b,trajectory = curve.encode_trajectory(pos)
    w_no_constraint = curve.decode_w(w_b).reshape(-1,2)

    # Create a grid for SDF visualization
    x = torch.linspace(min(x_min,y_min),min(x_max,y_max), 50)
    y = torch.linspace(min(x_min,y_min),min(x_max,y_max), 50)
    x, y = torch.meshgrid(x, y)
    p = torch.stack([x.flatten(), y.flatten()], dim=1).to(device)
    
    dist, grad, t = curve.sdf_batch(p.reshape(-1,2), w_b)
    dist_np, grad_np = dist.cpu().numpy(), grad.cpu().numpy()

    # Dynamical system
    p_next,vec_field = dynamical_system_single_step(curve,p,w_b)
    vec_field_np = vec_field.cpu().numpy()

    # Create figure for plotting
    fig, axes = plt.subplots(1,3,figsize=(18, 6))
    for ax in axes:
        ax.set_xlim(min(x_min,y_min).item(),min(x_max,y_max).item())
        ax.set_ylim(min(x_min,y_min).item(),min(x_max,y_max).item())
        ax.set_aspect('equal')
        ax.axis('square')

    # Plot reconstructed trajectory
    axes[0].plot(trajectory[:,0].cpu().numpy(),trajectory[:,1].cpu().numpy(),linewidth=2, c='black')
    # Plot control points
    axes[0].plot(w_no_constraint[:,0].cpu().numpy(),w_no_constraint[:,1].cpu().numpy(),"o-",markersize=3,color="#1f77b4")
    # Plot original data
    axes[0].plot(pos[:,0].cpu().numpy(),pos[:,1].cpu().numpy(),"x",markersize=3,color="red")   # original data
    axes[0].set_aspect('equal')  # Ensure equal scaling

    axes[1].plot(trajectory[:,0].cpu().numpy(),trajectory[:,1].cpu().numpy(),linewidth=2, c='black')
    axes[1].contourf(x.cpu().numpy(), y.cpu().numpy(), dist_np.reshape(50,50), cmap='coolwarm')
    axes[1].set_aspect('equal')

    axes[2].plot(trajectory[:,0].cpu().numpy(),trajectory[:,1].cpu().numpy(),linewidth=2, c='black')
    axes[2].streamplot(
        x.cpu().numpy()[::3, ::3].T,
        y.cpu().numpy()[::3, ::3].T,
        vec_field_np[:,0].reshape(50,50)[::3, ::3].T,
        vec_field_np[:,1].reshape(50,50)[::3, ::3].T,
        color="darkgray",
        linewidth=1,
        density=1.0,
    )

    axes[2].set_aspect('equal')
    plt.show()