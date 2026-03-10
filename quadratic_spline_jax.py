# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yiming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import math

import jax.numpy as jnp


def _normalize(x, axis=-1, eps=1e-8):
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / jnp.maximum(norm, eps)


class QuadraticSpline:
    def __init__(self, nbFct=3, nbSeg=10, nbDim=2, device="cuda"):
        self.device = device
        self.nbFct = nbFct
        if nbFct != 3:
            raise ValueError("Only quadratic splines are supported!")
        self.nbSeg = nbSeg
        self.nbDim = nbDim
        self.dtype = jnp.float32
        self.BC = self._compute_BC()
        self.M = jnp.kron(self.BC, self.BC)

    def binomial(self, n, i):
        if n >= 0 and i >= 0:
            return math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
        return 0

    def block_diag(self, A, B):
        out = jnp.zeros(
            (A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]), dtype=self.dtype
        )
        out = out.at[: A.shape[0], : A.shape[1]].set(A)
        out = out.at[A.shape[0] :, A.shape[1] :].set(B)
        return out

    # basis and constrained matrix
    def _compute_BC(self):
        B0 = jnp.zeros((self.nbFct, self.nbFct), dtype=self.dtype)
        for n in range(1, self.nbFct + 1):
            for i in range(1, self.nbFct + 1):
                value = (
                    (-1) ** (self.nbFct - i - n)
                    * (-self.binomial(self.nbFct - 1, i - 1))
                    * self.binomial(
                        self.nbFct - 1 - (i - 1),
                        self.nbFct - 1 - (n - 1) - (i - 1),
                    )
                )
                B0 = B0.at[self.nbFct - i, n - 1].set(value)
        B = jnp.kron(jnp.eye(self.nbSeg, dtype=self.dtype), B0)

        C0 = jnp.array([[1.0], [1.0], [2.0]], dtype=self.dtype)
        C = jnp.eye(self.nbFct - 1, dtype=self.dtype)
        for _ in range(self.nbSeg - 1):
            C = self.block_diag(C, C0)
        C = self.block_diag(C, jnp.eye(self.nbFct - 2, dtype=self.dtype))
        C = C.at[1:-1:6, 1].set(1)
        C = C.at[4:-1:6, 1].set(-1)
        idx = 4
        for n in range(self.nbSeg - 2):
            C = C.at[idx:-1:6, n + 2].set(2)
            C = C.at[idx + 3 : -1 : 6, n + 2].set(-2)
            idx += 3

        self.C = C
        return B @ C

    def computePsiList1D(self, t):
        t = jnp.asarray(t, dtype=self.dtype)
        p_float = jnp.arange(self.nbFct, dtype=self.dtype)
        p_int = jnp.arange(self.nbFct, dtype=jnp.int32)

        phi = jnp.zeros((t.shape[0], self.BC.shape[1]), dtype=self.dtype)
        dphi = jnp.zeros_like(phi)

        for k in range(t.shape[0]):
            tt = jnp.mod(t[k], 1.0 / self.nbSeg) * self.nbSeg
            idx_float = jnp.round(t[k] * self.nbSeg - tt)
            idx = idx_float.astype(jnp.int32)

            tt = jnp.where(idx < 0, tt + idx.astype(self.dtype), tt)
            idx = jnp.where(idx < 0, 0, idx)
            tt = jnp.where(
                idx > (self.nbSeg - 1),
                tt + (idx - (self.nbSeg - 1)).astype(self.dtype),
                tt,
            )
            idx = jnp.where(idx > (self.nbSeg - 1), self.nbSeg - 1, idx)

            T = tt**p_float
            dT = jnp.zeros((self.nbFct,), dtype=self.dtype)
            dT = dT.at[1:].set(p_float[1:] * (tt ** (p_float[1:] - 1)) * self.nbSeg)
            idl = (idx * self.nbFct + p_int).astype(jnp.int32)

            phi = phi.at[k, :].set(T @ self.BC[idl, :])
            dphi = dphi.at[k, :].set(dT @ self.BC[idl, :])

        Psi = jnp.kron(phi, jnp.eye(self.nbDim, dtype=self.dtype))
        dPsi = jnp.kron(dphi, jnp.eye(self.nbDim, dtype=self.dtype))
        return Psi, dPsi, phi

    def encode_trajectory(self, data):
        data = jnp.asarray(data, dtype=self.dtype)
        N = data.shape[0]
        t = jnp.linspace(0, 1, N, dtype=self.dtype)
        Psi, dPsi, phi = self.computePsiList1D(t)
        w = jnp.linalg.pinv(Psi) @ data.reshape(-1)
        trajectory = (Psi @ w).reshape(N, -1)
        return w, trajectory

    def encode_trajectory_given_w(self, w, nPoints=200):
        w = jnp.asarray(w, dtype=self.dtype)
        t = jnp.linspace(0, 1, nPoints, dtype=self.dtype)
        Psi, dPsi, phi = self.computePsiList1D(t)
        trajectory = (Psi @ w).reshape(nPoints, -1)
        return trajectory

    @staticmethod
    def quadratic_bezier_curve(t, control_pts):
        P0, P1, P2 = control_pts
        t = jnp.asarray(t)[..., None]
        return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2

    @staticmethod
    def quadratic_bezier_curve_grad(t, control_pts):
        P0, P1, P2 = control_pts
        t = jnp.asarray(t)[..., None]
        return 2 * (1 - t) * (P1 - P0) + 2 * t * (P2 - P1)

    def decode_w(self, w):
        w = jnp.asarray(w, dtype=self.dtype)
        # decode w to size (nbFct*nbSeg, nbDim)
        w_decode = jnp.kron(self.C, jnp.eye(self.nbDim, dtype=self.dtype)) @ w
        w_decode = w_decode.reshape(self.nbSeg, self.nbFct, self.nbDim)
        return w_decode

    def solve_cubic(self, a, b, c):
        """Find real roots for x^3 + a x^2 + b x + c = 0, vectorized."""
        a = jnp.asarray(a, dtype=self.dtype)
        b = jnp.asarray(b, dtype=self.dtype)
        c = jnp.asarray(c, dtype=self.dtype)
        a, b, c = jnp.broadcast_arrays(a, b, c)

        p = b - (a**2) / 3.0
        q = a * (2 * (a**2) - 9 * b) / 27.0 + c
        R = (0.5 * q) ** 2 + (p / 3.0) ** 3

        # Case where R >= 0: one real root (repeated to keep shape [*, 3]).
        s = jnp.sqrt(jnp.maximum(R, 0.0))
        s1 = -0.5 * q + s
        s2 = -0.5 * q - s
        v = jnp.cbrt(s1)
        w = jnp.cbrt(s2)
        real_root = v + w - a / 3.0
        roots_real = jnp.repeat(real_root[..., None], 3, axis=-1)

        # Case where R < 0: three distinct real roots.
        eps = jnp.asarray(1e-12, dtype=self.dtype)
        safe_p = jnp.where(jnp.abs(p) < eps, -eps, p)
        S = 2.0 * jnp.sqrt(jnp.maximum(-safe_p / 3.0, 0.0))
        acos_arg = (
            1.5
            * q
            / safe_p
            * jnp.sqrt(jnp.maximum(-3.0 / safe_p, 0.0))
        )
        acos_arg = jnp.clip(acos_arg, -1.0, 1.0)
        theta = jnp.arccos(acos_arg)
        angles = (theta[..., None] + 2.0 * jnp.arange(3, dtype=self.dtype) * jnp.pi) / 3.0
        roots_complex = S[..., None] * jnp.cos(angles) - a[..., None] / 3.0

        mask_real = R >= 0.0
        return jnp.where(mask_real[..., None], roots_real, roots_complex)

    def quadratic_bezier_curve_sdf(self, W, p):
        """Find distance from points to one quadratic bezier segment."""
        W = jnp.asarray(W, dtype=self.dtype)
        p = jnp.asarray(p, dtype=self.dtype)
        P0, P1, P2 = W
        d = P0 - p
        p1 = 2 * (P1 - P0)
        p2 = P0 - 2 * P1 + P2

        B = 1.5 * jnp.sum(p1 * p2) * jnp.ones((p.shape[0],), dtype=self.dtype)
        C = jnp.sum(d * p2, axis=-1) + 0.5 * jnp.sum(p1 * p1)
        D = 0.5 * jnp.sum(d * p1, axis=-1)
        coefs = jnp.stack([B, C, D], axis=-1)
        a = jnp.sum(p2 * p2)

        coefs = coefs / (a + 1e-8)
        tmin = jnp.clip(self.solve_cubic(coefs[:, 0], coefs[:, 1], coefs[:, 2]), 0.0, 1.0)
        tmin = tmin[:, :, None]
        p_curve = P0 + (p1 + p2 * tmin) * tmin

        distances = jnp.linalg.norm(p_curve - p[:, None, :], axis=-1)
        best_root_idx = jnp.argmin(distances, axis=-1)
        p_curve_closest = jnp.take_along_axis(
            p_curve, best_root_idx[:, None, None], axis=1
        ).squeeze(1)
        grad = p_curve_closest - p
        distance = jnp.take_along_axis(distances, best_root_idx[:, None], axis=-1).squeeze(-1)
        tmin = jnp.take_along_axis(tmin.squeeze(-1), best_root_idx[:, None], axis=-1).squeeze(-1)

        return distance, grad, tmin

    def quadractic_bezier_curve_batch(self, t_batch, W):
        t_batch = jnp.asarray(t_batch, dtype=self.dtype)
        W = jnp.asarray(W, dtype=self.dtype)
        w_decode = self.decode_w(W).reshape(self.nbFct * self.nbSeg, -1)
        t_int = t_batch.astype(jnp.int32)
        t_int = jnp.where(t_int == w_decode.shape[0] - 2, t_int - 1, t_int)
        ctrl_points = (w_decode[t_int], w_decode[t_int + 1], w_decode[t_int + 2])
        curve = self.quadratic_bezier_curve(t_batch - t_int, ctrl_points)
        return curve

    def quadractic_bezier_curve_grad_batch(self, t_batch, W):
        t_batch = jnp.asarray(t_batch, dtype=self.dtype)
        W = jnp.asarray(W, dtype=self.dtype)
        w_decode = self.decode_w(W).reshape(self.nbFct * self.nbSeg, -1)
        t_int = t_batch.astype(jnp.int32)
        t_int = jnp.where(t_int == w_decode.shape[0] - 2, t_int - 1, t_int)
        ctrl_points = (w_decode[t_int], w_decode[t_int + 1], w_decode[t_int + 2])
        curve_grad = self.quadratic_bezier_curve_grad(t_batch - t_int, ctrl_points)
        mask = t_batch > (self.nbSeg - 1) * self.nbFct + 0.99
        curve_grad = jnp.where(mask[:, None], jnp.zeros_like(curve_grad), curve_grad)
        return curve_grad

    def quadratic_bezier_curve_sdf_batch(self, W, p):
        """Find distance from points to batched quadratic bezier segments."""
        W = jnp.asarray(W, dtype=self.dtype)
        p = jnp.asarray(p, dtype=self.dtype)

        P0, P1, P2 = W[:, 0], W[:, 1], W[:, 2]  # each (B, 2)
        d = P0[:, None, :] - p[None, :, :]  # (B, N, 2)
        p1 = 2 * (P1 - P0)  # (B, 2)
        p2 = P0 - 2 * P1 + P2  # (B, 2)

        B_coeff = 1.5 * jnp.sum(p1 * p2, axis=-1, keepdims=True)  # (B, 1)
        C_coeff = jnp.sum(d * p2[:, None, :], axis=-1) + 0.5 * jnp.sum(
            p1[:, None, :] * p1[:, None, :], axis=-1
        )  # (B, N)
        D_coeff = 0.5 * jnp.sum(d * p1[:, None, :], axis=-1)  # (B, N)

        a = jnp.sum(p2 * p2, axis=-1)  # (B,)
        inv_a = 1.0 / (a[:, None, None] + 1e-8)  # (B, 1, 1)

        coefs = jnp.stack(
            [
                jnp.broadcast_to(B_coeff, C_coeff.shape),
                C_coeff,
                D_coeff,
            ],
            axis=-1,
        ) * inv_a  # (B, N, 3)

        B, N = coefs.shape[:2]
        coefs_flat = coefs.reshape(-1, 3)  # (B*N, 3)
        t_candidates = jnp.clip(
            self.solve_cubic(coefs_flat[:, 0], coefs_flat[:, 1], coefs_flat[:, 2]),
            0.0,
            1.0,
        )  # (B*N, 3)
        t_candidates = t_candidates.reshape(B, N, 3)  # (B, N, 3)

        t = t_candidates[:, :, :, None]  # (B, N, 3, 1)
        one_minus_t = 1 - t
        term0 = one_minus_t * one_minus_t * P0[:, None, None, :]  # (B, N, 3, 2)
        term1 = 2 * one_minus_t * t * P1[:, None, None, :]
        term2 = t * t * P2[:, None, None, :]
        p_curve = term0 + term1 + term2  # (B, N, 3, 2)

        distances = jnp.linalg.norm(p_curve - p[None, :, None, :], axis=-1)  # (B, N, 3)
        min_dist = jnp.min(distances, axis=-1)  # (B, N)
        best_idx = jnp.argmin(distances, axis=-1)  # (B, N)

        tmin = jnp.take_along_axis(t_candidates, best_idx[:, :, None], axis=-1).squeeze(-1)
        tmin = tmin + 3.0 * jnp.arange(B, dtype=self.dtype)[:, None]
        p_closest = jnp.take_along_axis(
            p_curve, best_idx[:, :, None, None], axis=2
        ).squeeze(2)
        grad = p_closest - p[None, :, :]  # (B, N, 2)
        return min_dist, grad, tmin

    def sdf(self, p, w):
        p = jnp.asarray(p, dtype=self.dtype)
        w_decode = self.decode_w(w)
        min_dist = []
        grad_list = []
        t_list = []
        for i, w_seg in enumerate(w_decode):
            dist, grad, tmin = self.quadratic_bezier_curve_sdf(w_seg, p)
            min_dist.append(dist)
            grad_list.append(grad)
            t_list.append(tmin + i)

        dist_stacked = jnp.stack(min_dist, axis=0)
        dist_idx = jnp.argmin(dist_stacked, axis=0)
        dist = jnp.take_along_axis(dist_stacked, dist_idx[None, :], axis=0).squeeze(0)

        grad_stacked = jnp.stack(grad_list, axis=0)
        grad = grad_stacked[dist_idx, jnp.arange(p.shape[0]), :]
        grad = _normalize(grad, axis=-1)

        t_stacked = jnp.stack(t_list, axis=0)
        t = t_stacked[dist_idx, jnp.arange(p.shape[0])]
        return dist, grad, t

    def sdf_batch(self, p, w):
        p = jnp.asarray(p, dtype=self.dtype)
        w_decode = self.decode_w(w)
        dist, grad, tmin = self.quadratic_bezier_curve_sdf_batch(w_decode, p)
        dist_idx = jnp.argmin(dist, axis=0)
        dist = jnp.take_along_axis(dist, dist_idx[None, :], axis=0).squeeze(0)
        grad = grad[dist_idx, jnp.arange(p.shape[0]), :]
        grad = _normalize(grad, axis=-1)
        t = tmin[dist_idx, jnp.arange(p.shape[0])]
        return dist, grad, t

    def multi_traj_sdf_batch(self, p, w_list):
        p = jnp.asarray(p, dtype=self.dtype)
        w_decode = jnp.concatenate([self.decode_w(w) for w in w_list], axis=0)
        dist, grad, tmin = self.quadratic_bezier_curve_sdf_batch(w_decode, p)

        dist_idx = jnp.argmin(dist, axis=0)
        dist = jnp.take_along_axis(dist, dist_idx[None, :], axis=0).squeeze(0)
        grad = grad[dist_idx, jnp.arange(p.shape[0]), :]
        grad = _normalize(grad, axis=-1)
        t_all = tmin[dist_idx, jnp.arange(p.shape[0])]

        t_seg = jnp.mod(t_all, self.nbSeg * self.nbFct)
        t_batch = (t_all // (self.nbSeg * self.nbFct)).astype(jnp.int32)
        w_decode_reshaped = w_decode.reshape(len(w_list), self.nbSeg * self.nbFct, -1)

        t_int = t_seg.astype(jnp.int32)
        t_int = jnp.where(t_int == w_decode_reshaped.shape[1] - 2, t_int - 1, t_int)
        ctrl_points = (
            w_decode_reshaped[t_batch, t_int],
            w_decode_reshaped[t_batch, t_int + 1],
            w_decode_reshaped[t_batch, t_int + 2],
        )
        curve = self.quadratic_bezier_curve(t_seg - t_int, ctrl_points)
        curve_grad = self.quadratic_bezier_curve_grad(t_seg - t_int, ctrl_points)
        return dist, grad, t_seg, curve, curve_grad

    def decode_time(self, t_batch):
        """
        Convert discontinuous time segments to continuous time.

        Args:
            t_batch: Array containing discontinuous time values.

        Returns:
            t_continuous: Array with continuous time values.
        """
        t_batch = jnp.asarray(t_batch, dtype=self.dtype)
        segment_index = jnp.floor_divide(t_batch, self.nbFct)
        local_time_within_segment = jnp.mod(t_batch, 1.0)
        t_continuous = segment_index + local_time_within_segment

        end_point_condition = t_batch == (self.nbSeg - 1) * self.nbFct + 1
        t_continuous = jnp.where(
            end_point_condition,
            jnp.asarray(self.nbSeg, dtype=t_batch.dtype),
            t_continuous,
        )
        return t_continuous


def dynamical_system_single_step(
    curve, p, w, lambda_dist=0.5, step_size=0.1, dist_threshold=0.0
):
    p = jnp.asarray(p, dtype=curve.dtype)
    w = jnp.asarray(w, dtype=curve.dtype)
    dist, grad, t = curve.sdf_batch(p, w)

    curve_grad = curve.quadractic_bezier_curve_grad_batch(t, w) * 0.1
    barrier = 1.0 / (1 + lambda_dist * jnp.abs(dist - dist_threshold) + 1e-6)
    mask = dist < dist_threshold
    grad = jnp.where(mask[:, None], grad * 0.5, grad)
    vec = curve_grad * barrier[:, None] + grad * (1 - barrier)[:, None]
    vec = _normalize(vec, axis=-1)
    p_next = p + vec * step_size
    return p_next, vec


def multi_traj_dynamical_system_single_step(curve, p, w_list, lambda_dist=0.5, step_size=0.1):
    p = jnp.asarray(p, dtype=curve.dtype)
    dist, grad, tmin, _, curve_grad = curve.multi_traj_sdf_batch(p, w_list)
    curve_grad = curve_grad * 0.1
    barrier = 1.0 / (1 + lambda_dist * dist + 1e-6)
    vec = curve_grad * barrier[:, None] + grad * (1 - barrier)[:, None]
    vec = _normalize(vec, axis=-1)
    p_next = p + vec * step_size
    return p_next, vec
