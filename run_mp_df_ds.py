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
    curve = QuadraticSpline(nbFct=3, nbSeg=2,device=device)

    # explicitely set control points
    w_b = torch.tensor([8.303, 8.592,  -10.000, 8.286,  0.500, -1.233, -10.000, -6.137],device=device)

    N = 100
    t= torch.linspace(0,1,N)
    Psi,dPsi,phi = curve.computePsiList1D(t)
    trajectory = (Psi.float() @ w_b.float()).reshape(N,-1)

    w_no_constraint = curve.decode_w(w_b).reshape(-1,2)

    # Create a grid for SDF visualization
    x_min,x_max = trajectory[:,0].min()-5.0,trajectory[:,0].max()+5.0
    y_min,y_max = trajectory[:,1].min()-5.0,trajectory[:,1].max()+5.0
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
        ax.set_axis_off()
        ax.set_xlim(min(x_min,y_min).item(),min(x_max,y_max).item())
        ax.set_ylim(min(x_min,y_min).item(),min(x_max,y_max).item())
        ax.set_aspect('equal')
        ax.axis('square')

    # Plot reconstructed trajectory
    axes[0].plot(trajectory[:,0].cpu().numpy(),trajectory[:,1].cpu().numpy(), "-", color="black",linewidth=3)
    # Plot control points
    axes[0].plot(w_no_constraint[:,0].cpu().numpy(),w_no_constraint[:,1].cpu().numpy(), "o-", c='blue',markersize=10)
    # Plot original data
    axes[0].set_aspect('equal')  # Ensure equal scaling

    axes[1].plot(trajectory[:,0].cpu().numpy(),trajectory[:,1].cpu().numpy(),linewidth=3, c='black')
    axes[1].contourf(x.cpu().numpy(), y.cpu().numpy(), dist_np.reshape(50,50), levels=np.linspace(0.0, np.max(dist_np), 7), cmap="coolwarm")
    axes[1].set_aspect('equal')
    # plot gradient

    pts = torch.tensor([[4.0, 4.0]],device=device)
    dist, grad, t = curve.sdf_batch(pts, w_b)
    curve_grad = curve.quadractic_bezier_curve_grad_batch(t,w_b)
    norm_curve_grad = curve_grad / torch.norm(curve_grad,dim=-1,keepdim=True)
    closest_point = curve.quadractic_bezier_curve_batch(t, w_b)

    axes[1].plot(pts[:,0].cpu().numpy(),pts[:,1].cpu().numpy(), "o", markersize=10,color="red")
    axes[1].plot(closest_point[:,0].cpu().numpy(), closest_point[:,1].cpu().numpy(), "o", markersize=10,color="orange")
    axes[1].quiver(
        pts[:,0].cpu().numpy(),
        pts[:,1].cpu().numpy(),
        grad[:,0].cpu().numpy(),
        grad[:,1].cpu().numpy(),
        color="red",
        width=0.01,
        pivot="tail",
        scale=6.00,
        # order=1,
    )
    # # plot tangent
    axes[1].quiver(
        closest_point[:,0].cpu().numpy(),
        closest_point[:,1].cpu().numpy(),
        norm_curve_grad[:,0].cpu().numpy(),
        norm_curve_grad[:,1].cpu().numpy(),
        color="orange",
        width=0.01,
        pivot="tail",
        scale=6.00,
        zorder=10,
    )
    axes[2].set_aspect('equal')

    axes[2].plot(trajectory[:,0].cpu().numpy(),trajectory[:,1].cpu().numpy(),linewidth=3, c='black')
    axes[2].streamplot(
        x.cpu().numpy()[::3, ::3].T,
        y.cpu().numpy()[::3, ::3].T,
        vec_field_np[:,0].reshape(50,50)[::3, ::3].T,
        vec_field_np[:,1].reshape(50,50)[::3, ::3].T,
        color="darkgray",
        linewidth=1,
        density=1.0,
    )
    p_list = []
    for i in range(500):
        p_next,_ = dynamical_system_single_step(curve,pts,w_b)
        p_list.append(p_next)
        pts = p_next
    p_list = torch.stack(p_list,dim=0).squeeze(1)
    axes[2].plot(p_list[:,0].cpu().numpy(),p_list[:,1].cpu().numpy(), "-", linewidth=2,color="red")
    axes[2].plot(p_list[0,0].cpu().numpy(),p_list[0,1].cpu().numpy(), "o", markersize=10,color="red")
    axes[2].plot(p_list[-1,0].cpu().numpy(),p_list[-1,1].cpu().numpy(), "o", markersize=10,color="black")
        
    plt.show()