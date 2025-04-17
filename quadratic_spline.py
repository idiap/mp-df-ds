# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yiming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import torch
import math

class QuadraticSpline:
    def __init__(self, nbFct=3, nbSeg=10,nbDim=2,device='cuda'):
        self.device = device
        self.nbFct = nbFct
        if nbFct !=3:
            raise ValueError("Only quadratic splines are supported!")
        self.nbSeg = nbSeg
        self.nbDim = nbDim
        self.BC = self._compute_BC()
        self.M = torch.kron(self.BC, self.BC)

    def binomial(self, n, i):
        if n >= 0 and i >= 0:
            return math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
        return 0

    def block_diag(self, A, B):
        out = torch.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]), dtype=A.dtype,device=self.device)
        out[:A.shape[0], :A.shape[1]] = A
        out[A.shape[0]:, A.shape[1]:] = B
        return out

    # basis and contrained matrix
    def _compute_BC(self):
        B0 = torch.zeros((self.nbFct, self.nbFct),device=self.device)
        for n in range(1, self.nbFct + 1):
            for i in range(1, self.nbFct + 1):
                B0[self.nbFct - i, n - 1] = (
                    (-1) ** (self.nbFct - i - n)
                    * (-self.binomial(self.nbFct - 1, i - 1))
                    * self.binomial(self.nbFct - 1 - (i - 1), self.nbFct - 1 - (n - 1) - (i - 1))
                )
        B = torch.kron(torch.eye(self.nbSeg,device=self.device), B0)

        C0 = torch.tensor([[1.0], [1.0], [2.0]],device=self.device)
        C = torch.eye(self.nbFct - 1,device=self.device)
        for _ in range(self.nbSeg - 1):
            C = self.block_diag(C, C0)
        C = self.block_diag(C, torch.eye(self.nbFct - 2,device=self.device))
        C[1:-1:6, 1] = 1
        C[4:-1:6, 1] = -1
        id = 4
        for n in range(self.nbSeg - 2):
            C[id:-1:6, n + 2] = 2
            C[id + 3:-1:6, n + 2] = -2
            id += 3

        self.C = C
        return B @ C

    def computePsiList1D(self, t):
        T = torch.zeros((1, self.nbFct),device=self.device)
        dT = torch.zeros((1, self.nbFct),device=self.device)
        phi = torch.zeros((len(t), self.BC.shape[1]),device=self.device)
        dphi = torch.zeros_like(phi,device=self.device)

        for k in range(len(t)):
            tt = torch.remainder(t[k], 1.0 / self.nbSeg) * self.nbSeg
            id_float = torch.round(t[k] * self.nbSeg - tt)
            id = id_float.long()

            if id < 0:
                tt = tt + id  # id is negative here
                id = 0
            if id > (self.nbSeg - 1):
                tt = tt + (id - (self.nbSeg - 1))
                id = self.nbSeg - 1

            p1 = torch.arange(self.nbFct)
            T[0, :] = tt ** p1
            dT[0, 1:] = p1[1:] * (tt ** (p1[1:] - 1)) * self.nbSeg
            idl = (id * self.nbFct + p1).long()

            phi[k, :] = T @ self.BC[idl, :]
            dphi[k, :] = dT @ self.BC[idl, :]

        Psi = torch.kron(phi, torch.eye(self.nbDim,device=self.device))
        dPsi = torch.kron(dphi, torch.eye(self.nbDim,device=self.device))
        return Psi, dPsi, phi

    def encode_trajectory(self,data):
        N= data.shape[0]
        t = torch.linspace(0, 1, N)
        Psi,dPsi,phi = self.computePsiList1D(t)
        w = torch.linalg.pinv(Psi) @ data.reshape(-1).float()
        trajectory = (Psi.float() @ w.float()).reshape(N,-1)
        return w,trajectory

    def encode_trajectory_given_w(self,w,nPoints = 200):
        t = torch.linspace(0, 1, nPoints)
        Psi,dPsi,phi = self.computePsiList1D(t)
        trajectory = (Psi.float() @ w.float()).reshape(nPoints,-1)
        return trajectory

    @staticmethod
    def quadratic_bezier_curve(t, control_pts):
        P0, P1, P2 = control_pts
        t = t.unsqueeze(-1)
        return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2
    
    @staticmethod
    def quadratic_bezier_curve_grad(t, control_pts):
        P0, P1, P2 = control_pts
        t = t.unsqueeze(-1)
        return 2 * (1 - t) * (P1 - P0) + 2 * t * (P2 - P1)

    def decode_w(self,w):

        # decode w to size (nbFct*nbSeg,nbDim) 
        w_decode = torch.kron(self.C, torch.eye(self.nbDim, device=self.device)).float() @ w.float()
        w_decode = w_decode.reshape(self.nbSeg,self.nbFct,self.nbDim) 

        return w_decode
    
    def solve_cubic(self,a, b, c):
        """Find the signed distance from a point to a quadratic bezier curve.
        Supports batch operations in PyTorch.
        """
        p = b - (a ** 2) / 3
        q = a * (2 * (a ** 2) - 9 * b) / 27 + c
        R = (0.5 * q) ** 2 + (p / 3) ** 3
        batch_size = a.shape[0] if a.dim() > 0 else 1
        results = torch.zeros((batch_size, 3), device=a.device, dtype=a.dtype)

        # Case where R >= 0
        mask_real = R >= 0.0
        if mask_real.any():
            s = torch.sqrt(R[mask_real])
            s1 = -0.5 * q[mask_real] + s
            s2 = -0.5 * q[mask_real] - s
            v = torch.sign(s1) * torch.abs(s1) ** (1 / 3)
            w = torch.sign(s2) * torch.abs(s2) ** (1 / 3)
            results[mask_real.squeeze()] = (v.unsqueeze(-1) + w.unsqueeze(-1) - a[mask_real].unsqueeze(-1) / 3).repeat(1, 3)
        
        # Case where R < 0 (3 real solutions)
        mask_complex = ~mask_real
        if mask_complex.any():
            S = 2 * torch.sqrt(-p[mask_complex] / 3)
            theta = torch.acos(1.5 * q[mask_complex] / p[mask_complex] * torch.sqrt(-3.0 / p[mask_complex]))
            angles = (theta.unsqueeze(-1) + 2 * torch.arange(3, device=a.device).float() * torch.pi) / 3
            results[mask_complex.squeeze()] = S.unsqueeze(-1) * torch.cos(angles) - a[mask_complex].unsqueeze(-1) / 3
        return results
    
    def quadratic_bezier_curve_sdf(self,W, p):
        """Find the signed distance from a point to a quadratic bezier curve.
        By finding the best tmin minimizing the distance
        between a point P(x,y) and each segment of the spline.
        
        Minimizing the distance from a point to a quadratic bezier curve requires solving a cubic equation
        and selecting the root giving the lowest distance.
        W:(3,2)
        p:(N,2)
        
        """
        P0, P1, P2 = W
        d = P0 - p
        p1 = 2 * (P1 - P0)
        p2 = P0 - 2 * P1 + P2

        B = 3 * 0.5 * torch.sum(p1 * p2, dim=-1)*torch.ones(len(p),device=self.device)
        C = torch.sum(d * p2, dim=-1) + 0.5 * torch.sum(p1 * p1, dim=-1)
        D = 0.5 * torch.sum(d * p1, dim=-1)
        coefs = torch.stack([B,C,D], dim=-1)
        a = torch.sum(p2 * p2, dim=-1)

        tmin = torch.zeros_like(d)
        coefs /= a.unsqueeze(-1)
        tmin = torch.clamp(self.solve_cubic(*coefs.T), 0.0, 1.0).reshape(-1, 3,1)
        p_curve = P0 + (p1 + p2 * tmin) * tmin
        
        distances = torch.norm(p_curve - p.unsqueeze(1), dim=-1)
        best_root_idx = torch.argmin(distances, dim=-1)
        p_curve_closest = p_curve[torch.arange(len(p)),best_root_idx,:]
        grad = p_curve_closest - p
        distance = distances.gather(-1, best_root_idx.unsqueeze(-1)).squeeze(-1)
        tmin = tmin[torch.arange(len(p)),best_root_idx,:]

        return distance, grad.squeeze(-1), tmin.squeeze(-1)

    def quadractic_bezier_curve_batch(self,t_batch,W):
        w_decode = self.decode_w(W).reshape(self.nbFct*self.nbSeg,-1)
        # Dynamical system
        t_int = t_batch.int()
        t_int[t_int== w_decode.shape[0] - 2] = t_int[t_int== w_decode.shape[0] - 2] - 1

        expand_wb = w_decode.unsqueeze(0).expand(t_batch.shape[0],-1,-1)
        ctrl_points = (expand_wb[torch.arange(t_batch.shape[0]),t_int],
                    expand_wb[torch.arange(t_batch.shape[0]),t_int+1],
                    expand_wb[torch.arange(t_batch.shape[0]),t_int+2])
        # Compute trajectory gradient
        curve = self.quadratic_bezier_curve(t_batch - t_int, ctrl_points)
        return curve

    def quadractic_bezier_curve_grad_batch(self,t_batch,W):
        w_decode = self.decode_w(W).reshape(self.nbFct*self.nbSeg,-1)
        # Dynamical system
        t_int = t_batch.int()
        # exit()
        t_int[t_int== w_decode.shape[0] - 2] = t_int[t_int== w_decode.shape[0] - 2] - 1

        expand_wb = w_decode.unsqueeze(0).expand(t_batch.shape[0],-1,-1)
        ctrl_points = (expand_wb[torch.arange(t_batch.shape[0]),t_int],
                    expand_wb[torch.arange(t_batch.shape[0]),t_int+1],
                    expand_wb[torch.arange(t_batch.shape[0]),t_int+2])
        # Compute trajectory gradient
        curve_grad = self.quadratic_bezier_curve_grad(t_batch - t_int, ctrl_points)
        # asign 0 velocity to the end of the trajectory 
        curve_grad[t_batch>(self.nbSeg-1)*self.nbFct+0.99] = 0
        return curve_grad

    
    def quadratic_bezier_curve_sdf_batch(self, W, p):
        """Find the signed distance from points to quadratic bezier curves.
        
        Args:
            W: Batched quadratic bezier control points with shape (B, 3, 2)
            p: Points to evaluate with shape (N, 2)
            
        Returns:
            distance: Shape (B, N)
            grad: Shape (B, N, 2)
            tmin: Shape (B, N)
        """
        # Split control points
        P0, P1, P2 = W.unbind(dim=1)  # each (B, 2)
        
        # Prepare for broadcasting
        d = P0.unsqueeze(1) - p.unsqueeze(0)  # (B, N, 2)
        p1 = 2 * (P1 - P0)  # (B, 2)
        p2 = P0 - 2 * P1 + P2  # (B, 2)
        
        # Compute cubic equation coefficients
        B_coeff = 1.5 * torch.sum(p1.unsqueeze(1) * p2.unsqueeze(1), dim=-1)  # (B, 1)
        C_coeff = (torch.sum(d * p2.unsqueeze(1), dim=-1) + 
                0.5 * torch.sum(p1.unsqueeze(1) * p1.unsqueeze(1), dim=-1))  # (B, N)
        D_coeff = 0.5 * torch.sum(d * p1.unsqueeze(1), dim=-1)  # (B, N)
        
        # Normalization factor
        a = torch.sum(p2 * p2, dim=-1)  # (B,)
        eps = 1e-8
        inv_a = 1.0 / (a.unsqueeze(-1).unsqueeze(-1) + eps)  # (B, 1, 1)
        # print(f"inv_a: {inv_a.shape}, B_coeff: {B_coeff.shape}, C_coeff: {C_coeff.shape}, D_coeff: {D_coeff.shape}")
        # Stack and normalize coefficients
        coefs = torch.stack([
            B_coeff.expand(-1, p.shape[0]), 
            C_coeff,
            D_coeff
        ], dim=-1) * inv_a  # (B, N, 3)
        
        # Solve cubic equation
        B, N = coefs.shape[:2]
        coefs_flat = coefs.reshape(-1, 3)  # (B*N, 3)
        t_candidates = torch.clamp(self.solve_cubic(*coefs_flat.T), 0.0, 1.0)  # (B*N, 3)
        t_candidates = t_candidates.reshape(B, N, 3)  # (B, N, 3)
        
        # CORRECTED Curve evaluation: P(t) = (1-t)^2*P0 + 2*(1-t)*t*P1 + t^2*P2
        t = t_candidates.unsqueeze(-1)  # (B, N, 3, 1)
        one_minus_t = 1 - t
        
        # Compute each term
        term0 = one_minus_t * one_minus_t * P0.unsqueeze(1).unsqueeze(2)  # (B, N, 3, 2)
        term1 = 2 * one_minus_t * t * P1.unsqueeze(1).unsqueeze(2)
        term2 = t * t * P2.unsqueeze(1).unsqueeze(2)
        
        p_curve = term0 + term1 + term2  # (B, N, 3, 2)
        
        # Compute distances to original points
        distances = torch.norm(p_curve - p.unsqueeze(0).unsqueeze(2), dim=-1)  # (B, N, 3)
        
        # Find best t for each point
        min_dist, best_idx = torch.min(distances, dim=-1)  # both (B, N)
        
        # Gather results
        batch_idx = torch.arange(B, device=W.device).view(B, 1, 1).expand(-1, N, 1)
        point_idx = torch.arange(N, device=W.device).view(1, N, 1).expand(B, -1, 1)
        tmin = t_candidates.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
        # Add segment index to tmin
        tmin = tmin + 3.0*torch.arange(B,device=W.device).view(B,1).expand(-1,N)
        p_closest = p_curve[batch_idx, point_idx, best_idx.unsqueeze(-1)].squeeze(-2)  # (B, N, 2)
        
        # Compute gradient (normal vector)
        grad = p_closest - p.unsqueeze(0)  # (B, N, 2)
        
        return min_dist, grad, tmin

    def sdf(self,p,w):
        w_decode = self.decode_w(w)
        min_dist = []
        grad_list = []
        t_list = []
        for i,w_seg in enumerate(w_decode):
            # print(w_seg.shape)
            dist,grad,tmin = self.quadratic_bezier_curve_sdf(w_seg, p)
            tmin += i
            grad.squeeze_(0)
            min_dist.append(dist)
            grad_list.append(grad.squeeze(0))
            t_list.append(tmin)

        dist,dist_idx = torch.min(torch.stack(min_dist,dim=0),dim=0)
        grad = torch.stack(grad_list,dim=0)
        grad = grad[dist_idx,torch.arange(len(p)),:]
        grad = torch.nn.functional.normalize(grad,dim=-1)
        t = torch.stack(t_list,dim=0)
        t = t[dist_idx,torch.arange(len(p))]
        return dist,grad,t
    
    def sdf_batch(self,p,w):
        w_decode = self.decode_w(w)
        dist,grad,tmin = self.quadratic_bezier_curve_sdf_batch(w_decode, p)
        dist,dist_idx = torch.min(dist,dim=0)
        grad = grad[dist_idx,torch.arange(len(p)),:]
        grad = torch.nn.functional.normalize(grad,dim=-1)
        t = tmin[dist_idx,torch.arange(len(p))]
        return dist,grad,t

    def multi_traj_sdf_batch(self,p,w_list):
        dist_list,grad_list,t_list = [],[],[]
        for w in w_list:
            dist,grad,t = self.sdf_batch(p,w)
            dist_list.append(dist)
            grad_list.append(grad)
            t_list.append(t)
        dist,dist_idx = torch.min(torch.stack(dist_list,dim=0),dim=0)
        grad = torch.stack(grad_list,dim=0)
        grad = grad[dist_idx,torch.arange(len(p)),:]
        grad = torch.nn.functional.normalize(grad,dim=-1)
        t = torch.stack(t_list,dim=0)
        t = t[dist_idx,torch.arange(len(p))]
        return dist,grad,t


def dynamical_system_single_step(curve,p,w,lambda_dist=0.5,step_size=0.1):
    dist,grad,t = curve.sdf_batch(p,w)

    curve_grad = curve.quadractic_bezier_curve_grad_batch(t,w)*0.1
    # Compute barrier function
    barrier = 1.0/(1 + lambda_dist*dist+1e-6)
    # Combine trajectory and gradient fields
    vec = curve_grad * barrier.unsqueeze(-1) + grad * (1-barrier).unsqueeze(-1)
    vec = torch.nn.functional.normalize(vec,dim=-1)

    p_next = p + vec*step_size
    return p_next, vec

def multi_traj_dynamical_system_single_step(curve,p,w_list,lambda_dist=0.5,step_size=0.1):

    p_next_list, vec_field_list = [],[]
    dist_list,grad_list,t_list = [],[],[]
    for w in w_list:
        dist,grad,t = curve.sdf_batch(p,w)
        dist_list.append(dist)
        grad_list.append(grad)
        t_list.append(t)
        curve_grad = curve.quadractic_bezier_curve_grad_batch(t,w)*0.1
        # Compute barrier function
        barrier = 1.0/(1 + lambda_dist*dist+1e-6)
        # Combine trajectory and gradient fields
        vec = curve_grad * barrier.unsqueeze(-1) + grad * (1-barrier).unsqueeze(-1)
        vec = torch.nn.functional.normalize(vec,dim=-1)
        p_next = p + vec*step_size
        p_next_list.append(p_next)
        vec_field_list.append(vec)

    dist,dist_idx = torch.min(torch.stack(dist_list,dim=0),dim=0)
    p_next = torch.stack(p_next_list,dim=0)
    p_next = p_next[dist_idx,torch.arange(len(p)),:]
    vec_field = torch.stack(vec_field_list,dim=0)
    vec_field = vec_field[dist_idx,torch.arange(len(p)),:]
    vec_field = torch.nn.functional.normalize(vec_field,dim=-1)
    return p_next, vec_field
