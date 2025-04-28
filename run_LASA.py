# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yiming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import torch
import matplotlib.pyplot as plt
from quadratic_spline import QuadraticSpline,multi_traj_dynamical_system_single_step
import pyLasaDataset as lasa


if __name__ == "__main__":
    device = 'cuda'
    curve = QuadraticSpline(nbFct=3, nbSeg=10,device=device)

    # Create a trajectory concatenating two letters
    data_set_names = ['Angle','BendedLine','CShape','DoubleBendedLine','GShape','JShape','JShape_2','Khamesh','LShape','Leaf_1','Leaf_2','Line','Multi_Models_1','Multi_Models_2','Multi_Models_3','Multi_Models_4','NShape','PShape','RShape','Saeghe','Sharpc','Sine','Snake','Spoon','Sshape','Trapezoid','WShape','Worm','Zshape','heee']
    for dataset_name in data_set_names:
        if dataset_name !='Leaf_2':
            continue
        lasa_data = getattr(lasa.DataSet, dataset_name)
        demos = lasa_data.demos
        w_list,trajectory_list = [],[]
        x_min,y_min,x_max,y_max = torch.inf,torch.inf,-torch.inf,-torch.inf
        fig, axes = plt.subplots(1,3,figsize=(18, 6))
        for demo in demos:
            pos = torch.from_numpy(demo.pos.T).to(device)
            x_pos_min, x_pos_max = pos[:, 0].min(), pos[:, 0].max()
            y_pos_min, y_pos_max = pos[:, 1].min(), pos[:, 1].max()
            x_min = min(x_min,x_pos_min)
            y_min = min(y_min,y_pos_min)
            x_max = max(x_max,x_pos_max)
            y_max = max(y_max,y_pos_max)

        for demo in demos:
            # normalize the trajectory
            pos = torch.from_numpy(demo.pos.T).to(device)
            pos_normalized = (pos - torch.tensor([x_min, y_min],device=device)) / torch.tensor([x_max - x_min, y_max - y_min],device=device)
            w_b,trajectory = curve.encode_trajectory(pos_normalized.reshape(-1,2))
            w_no_constraint = curve.decode_w(w_b).reshape(-1,2)
            w_list.append(w_b)
            trajectory_list.append(trajectory)

        for ax in axes:
            ax.set_xlim(-0.5,1.5)
            ax.set_ylim(-0.5,1.5)
            ax.set_aspect('equal')
            ax.axis('square')
        for trajectory in trajectory_list:
            for ax in axes:
                ax.plot(trajectory[:,0].cpu().numpy(),trajectory[:,1].cpu().numpy(),linewidth=2, c='b',alpha=1.0)
        # Create a grid for visualization
        x = torch.linspace(-0.5,1.5, 50)
        y = torch.linspace(-0.5,1.5, 50)
        x, y = torch.meshgrid(x, y)
        p = torch.stack([x.flatten(), y.flatten()], dim=1).to(device)
        # Create figure for plotting

        # Distance Field
        dist, grad, t = curve.multi_traj_sdf_batch(p,w_list)

        # Dynamical system
        p_next,vec_field = multi_traj_dynamical_system_single_step(curve,p,w_list)

        dist_np,vec_field_np = dist.cpu().numpy(),vec_field.cpu().numpy()
        axes[1].plot(trajectory[:,0].cpu().numpy(),trajectory[:,1].cpu().numpy(),linewidth=2, c='black')
        axes[1].contourf(x.cpu().numpy(), y.cpu().numpy(), dist_np.reshape(50,50), cmap='coolwarm')
        axes[1].set_aspect('equal')

        # specific point shooting
        pts = torch.tensor([[-0.5,-0.5],[-0.5,1.5],[1.5,-0.5],[1.5,1.5]],device=device)
        p_list = []
        for i in range(500):
            p_next,_ = multi_traj_dynamical_system_single_step(curve,pts,w_list,step_size=0.05)
            p_list.append(p_next)
            pts = p_next
        p_list = torch.stack(p_list,dim=0).squeeze(1)
        ax.plot(p_list[:,:,0].cpu().numpy(),p_list[:,:,1].cpu().numpy(),c='magenta',linewidth=3)

        axes[2].plot(p_list[-1,:,0].cpu().numpy(),p_list[-1,:,1].cpu().numpy(), "o", markersize=10,color="black")

        axes[2].contourf(x.cpu().numpy(),y.cpu().numpy(),dist_np.reshape(50,50),cmap='Wistia',levels=10,alpha=0.5)
        axes[2].contour(x.cpu().numpy(),y.cpu().numpy(),dist_np.reshape(50,50),levels=10,colors='white',linewidths=1)
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