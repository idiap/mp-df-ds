# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yiming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from quadratic_spline_jax import QuadraticSpline, dynamical_system_single_step


if __name__ == "__main__":
    print("Construct spline...")
    curve = QuadraticSpline(nbFct=3, nbSeg=2, device="cpu")

    # explicitly set control points
    w_b = jnp.array(
        [8.303, 8.592, -10.000, 8.286, 0.500, -1.233, -10.000, -6.137],
        dtype=jnp.float32,
    )

    N = 40 #Number of points in the trajectory
    t = jnp.linspace(0, 1, N, dtype=jnp.float32)
    Psi, dPsi, phi = curve.computePsiList1D(t)
    trajectory = (Psi @ w_b).reshape(N, -1)

    w_no_constraint = curve.decode_w(w_b).reshape(-1, 2)

    # Create a grid for SDF visualization
    print("Create grid for SDF visualization...")
    sz = 40 #size of the grid
    x_min, x_max = trajectory[:, 0].min() - 5.0, trajectory[:, 0].max() + 5.0
    y_min, y_max = trajectory[:, 1].min() - 5.0, trajectory[:, 1].max() + 5.0
    x = jnp.linspace(jnp.minimum(x_min, y_min), jnp.minimum(x_max, y_max), sz)
    y = jnp.linspace(jnp.minimum(x_min, y_min), jnp.minimum(x_max, y_max), sz)
    x, y = jnp.meshgrid(x, y, indexing="ij")
    p = jnp.stack([x.reshape(-1), y.reshape(-1)], axis=1)

    dist, grad, t = curve.sdf_batch(p.reshape(-1, 2), w_b)
    dist_np, grad_np = np.asarray(dist), np.asarray(grad)

    # Dynamical system
    print("Compute vector field...")
    p_next, vec_field = dynamical_system_single_step(curve, p, w_b)
    vec_field_np = np.asarray(vec_field)

    # Create figure for plotting
    print("Plot result...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x_low = float(jnp.minimum(x_min, y_min))
    x_high = float(jnp.minimum(x_max, y_max))
    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(x_low, x_high)
        ax.set_aspect("equal")
        ax.axis("square")

    trajectory_np = np.asarray(trajectory)
    w_no_constraint_np = np.asarray(w_no_constraint)
    p_np = np.asarray(p)
    x_np, y_np = np.asarray(x), np.asarray(y)

    # Plot reconstructed trajectory
    axes[0].plot(trajectory_np[:, 0], trajectory_np[:, 1], "-", color="black", linewidth=3)
    # Plot control points
    axes[0].plot(
        w_no_constraint_np[:, 0],
        w_no_constraint_np[:, 1],
        "o-",
        c="blue",
        markersize=10,
    )
    axes[0].set_aspect("equal")

    axes[1].plot(trajectory_np[:, 0], trajectory_np[:, 1], linewidth=3, c="black")
    axes[1].contourf(
        x_np,
        y_np,
        dist_np.reshape(sz, sz),
        levels=np.linspace(0.0, np.max(dist_np), 7),
        cmap="coolwarm",
    )
    axes[1].set_aspect("equal")

    # Plot gradient at selected points
    pts = jnp.array([[4.0, 4.0]], dtype=jnp.float32)
    dist, grad, t = curve.sdf_batch(pts, w_b)
    curve_grad = curve.quadractic_bezier_curve_grad_batch(t, w_b)
    norm_curve_grad = curve_grad / jnp.maximum(
        jnp.linalg.norm(curve_grad, axis=-1, keepdims=True), 1e-8
    )
    closest_point = curve.quadractic_bezier_curve_batch(t, w_b)

    pts_np = np.asarray(pts)
    grad_np_pt = np.asarray(grad)
    closest_point_np = np.asarray(closest_point)
    norm_curve_grad_np = np.asarray(norm_curve_grad)

    axes[1].plot(pts_np[:, 0], pts_np[:, 1], "o", markersize=10, color="red")
    axes[1].plot(
        closest_point_np[:, 0],
        closest_point_np[:, 1],
        "o",
        markersize=10,
        color="orange",
    )
    axes[1].quiver(
        pts_np[:, 0],
        pts_np[:, 1],
        grad_np_pt[:, 0],
        grad_np_pt[:, 1],
        color="red",
        width=0.01,
        pivot="tail",
        scale=6.0,
    )
    axes[1].quiver(
        closest_point_np[:, 0],
        closest_point_np[:, 1],
        norm_curve_grad_np[:, 0],
        norm_curve_grad_np[:, 1],
        color="orange",
        width=0.01,
        pivot="tail",
        scale=6.0,
        zorder=10,
    )
    axes[2].set_aspect("equal")

    axes[2].plot(trajectory_np[:, 0], trajectory_np[:, 1], linewidth=3, c="black")
    axes[2].streamplot(
        x_np[::3, ::3].T,
        y_np[::3, ::3].T,
        vec_field_np[:, 0].reshape(sz, sz)[::3, ::3].T,
        vec_field_np[:, 1].reshape(sz, sz)[::3, ::3].T,
        color="darkgray",
        linewidth=1,
        density=1.0,
    )

    @jax.jit
    def run_simulation(pts_init, w):
        def scan_step(pts, _):
            p_next, _ = dynamical_system_single_step(curve, pts, w)
            return p_next, p_next

        _, p_list = jax.lax.scan(scan_step, pts_init, None, length=500)
        return p_list

    p_list = run_simulation(pts, w_b).transpose(1, 0, 2)

    for points in np.asarray(p_list):
        axes[2].plot(points[:, 0], points[:, 1], "-", linewidth=2, color="red")
        axes[2].plot(points[0, 0], points[0, 1], "o", markersize=10, color="red")
        axes[2].plot(points[-1, 0], points[-1, 1], "o", markersize=10, color="black")

    plt.show()
