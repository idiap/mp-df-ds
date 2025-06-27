# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yiming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import torch
import matplotlib.pyplot as plt
from quadratic_spline import QuadraticSpline,multi_traj_dynamical_system_single_step,multi_traj_dynamical_system_single_step
import numpy as np
import os
    
if __name__ == "__main__":
    device = 'cuda'
    curve = QuadraticSpline(nbFct=3, nbSeg=10,device=device)

    # Create a trajectory concatenating two letters
    data_dir = os.path.join(os.path.dirname(__file__), "data/")
    CShape = np.load(data_dir + "S.npy", allow_pickle="True")
    CShape = torch.from_numpy(CShape).to(device)
    pos_list,vel_list,acc_list = CShape[:5,:,:2],CShape[:5,:,2:4],CShape[:5,:,4:]
    # Encode trajectory
    x_min = pos_list[:,:,0].min()-5.0
    x_max = pos_list[:,:,0].max()+5.0
    y_min = pos_list[:,:,1].min()-5.0
    y_max = pos_list[:,:,1].max()+5.0

    # Create a grid for visualization
    x = torch.linspace(min(x_min,y_min),max(x_max,y_max), 50)
    y = torch.linspace(min(x_min,y_min),max(x_max,y_max), 50)
    x, y = torch.meshgrid(x, y)
    p = torch.stack([x.flatten(), y.flatten()], dim=1).to(device)
    # Create figure for plotting

    fig, axes = plt.subplots(1,3,figsize=(18, 6))
    for ax in axes:
        ax.set_xlim(min(x_min,y_min).item(),max(x_max,y_max).item())
        ax.set_ylim(min(x_min,y_min).item(),max(x_max,y_max).item())
        ax.set_aspect('equal')
        ax.axis('square')

    w_list = []
    for pos in pos_list:
    # Encode trajectory
        w_b,trajectory = curve.encode_trajectory(pos.reshape(-1,2))
        w_no_constraint = curve.decode_w(w_b).reshape(-1,2)
        w_list.append(w_b)
            # Plot reconstructed trajectory
        for ax in axes:
            ax.plot(trajectory[:,0].cpu().numpy(),trajectory[:,1].cpu().numpy(),linewidth=3, c='black',alpha=1.0)

    # Mean trajectory 
    # (Note: This mean trajectory is also represented by the quadratic spline, so you can also compute its distance field and corresponding vector field. In the following example, we show how to combine mutilple distance fields and vector fields instead of simply using the mean trajectory.)
    w_mean = torch.mean(torch.stack(w_list,dim=0),dim=0)
    w_mean_no_constraint = curve.decode_w(w_mean).reshape(-1,2)
    mean_trajectory = curve.encode_trajectory_given_w(w_mean)
    axes[0].plot(mean_trajectory[:,0].cpu().numpy(),mean_trajectory[:,1].cpu().numpy(),linewidth=5, c='grey',alpha=1.0)
    axes[0].plot(w_mean_no_constraint[:,0].cpu().numpy(),w_mean_no_constraint[:,1].cpu().numpy(),"o-",markersize=5,color="#1f77b4")

    # Distance Field
    dist, grad, t ,_,_ = curve.multi_traj_sdf_batch(p,w_list)
    # Dynamical system
    p_next,vec_field = multi_traj_dynamical_system_single_step(curve,p,w_list)

    dist_np,vec_field_np = dist.cpu().numpy(),vec_field.cpu().numpy()
    axes[1].contourf(x.cpu().numpy(), y.cpu().numpy(), dist_np.reshape(50,50), cmap='coolwarm')
    axes[1].set_aspect('equal')

    # specific point shooting
    pts = torch.tensor([[5.0, 5.0]],device=device)
    p_list = []
    for i in range(500):
        p_next,_ = multi_traj_dynamical_system_single_step(curve,pts,w_list)
        p_list.append(p_next)
        pts = p_next
    p_list = torch.stack(p_list,dim=0).squeeze(1)
    
    axes[2].plot(p_list[:,0].cpu().numpy(),p_list[:,1].cpu().numpy(), "-", linewidth=2,color="red")
    axes[2].plot(p_list[0,0].cpu().numpy(),p_list[0,1].cpu().numpy(), "o", markersize=10,color="red")
    axes[2].plot(p_list[-1,0].cpu().numpy(),p_list[-1,1].cpu().numpy(), "o", markersize=10,color="black")

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