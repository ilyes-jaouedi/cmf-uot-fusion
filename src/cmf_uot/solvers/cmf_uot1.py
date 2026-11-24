import numpy as np
import time

class TransportOptimizerUnifiedGCD:
    """
    Optimizes the UOT-regularized source localization problem using a Greedy
    Block-Coordinate Descent algorithm on the (P, b) variable pair.

    This algorithm decouples the acoustic power 'b' from the assignment plan 'P',
    reframing the problem as a pure assignment task.
    """
    def __init__(self, C, a, sigma, G, lambda_=1.0, mu1=1.0, mu2=10.0):
        self.m = C.shape[0]  # num_grid_points
        self.n = C.shape[1]  # num_camera_detections
        
        # --- Model Parameters ---
        self.lambda_ = lambda_
        self.mu1 = mu1
        self.mu2 = mu2  
        
        # --- Problem Data ---
        self.C = C
        self.G = G
        # It is assumed the diagonal of sigma (noise variance) is handled elsewhere
        # or that sigma is the noise-free covariance matrix.
        self.sigma = sigma - np.diag(np.diag(sigma))
        self.a = a 
        
        # --- State Variables ---
        self.P = np.zeros((self.m, self.n))
        self.b = np.zeros(self.m)

        self.support = np.zeros(self.m, dtype=bool)
    
    def loss_function_split(self):
        """Calculates the loss for the new objective function with split variables."""
        p_marginal = self.P.sum(axis=1)
        
        cmf_R = self.G @ (self.b[:, np.newaxis] * self.G.conj().T) - self.sigma
        term1_cmf = (self.lambda_ / 2) * np.linalg.norm(cmf_R, ord='fro') ** 2
        
        term2_cost = np.sum(self.C * self.P)
        
        residual_a = self.P.sum(axis=0) - self.a
        term3_mu1 = (self.mu1 / 2) * np.linalg.norm(residual_a, ord=2) ** 2
        
        residual_b = p_marginal - self.b
        term4_mu2 = (self.mu2 / 2) * np.linalg.norm(residual_b, ord=2) ** 2
        
        return term1_cmf + term2_cost + term3_mu1 + term4_mu2

    def _compute_gradient_P(self, P, b):
        """Computes the gradient with respect to P."""
        col_sums = P.sum(axis=0)
        row_sums = P.sum(axis=1)
        mu1_term = self.mu1 * (col_sums - self.a)
        mu2_term = self.mu2 * (row_sums - b)[:, np.newaxis]
        return self.C + mu1_term + mu2_term

    def _compute_gradient_b(self, b, P_marginal):
        """Computes the gradient with respect to b."""
        R = self.G @ (b[:, np.newaxis] * self.G.conj().T) - self.sigma
        GR = R @ self.G
        cmf_grad = self.lambda_ * np.einsum('mc,mc->c', self.G.conj(), GR).real
        reg_grad = -self.mu2 * (P_marginal - b)
        return cmf_grad + reg_grad

    def _solve_all_2d_subproblems(self, grad_P, grad_b, P, b, H_pp, H_pb, H_bb_vec,inv_det_H):
        """
        Vectorized analytical solver for the constrained 2D quadratic subproblems.
        [FINAL CORRECTED VERSION]
        """
        # --- Start with arrays that are already (m, n) ---
        g_P = grad_P
        L_P = -P
        L_B = -b[:, np.newaxis]

        # --- Explicitly broadcast all (m,) or (m, 1) arrays to the full (m, n) shape ---
        # This is the key fix that prevents all subsequent indexing errors.
        g_B_full = np.broadcast_to(grad_b[:, np.newaxis], (self.m, self.n))
        L_B_full = np.broadcast_to(L_B, (self.m, self.n))
        H_BB_full = np.broadcast_to(H_bb_vec[:, np.newaxis], (self.m, self.n))

        # --- 1. Unconstrained Solution (using full-sized arrays) ---
        
        alpha_star_P = inv_det_H * (-H_BB_full * g_P + H_pb * g_B_full)
        alpha_star_B = inv_det_H * ( H_pb * g_P - H_pp * g_B_full)
        
        # --- 2. Constrained Solution via KKT case analysis ---
        alpha_p_opt = alpha_star_P.copy()
        alpha_b_opt = alpha_star_B.copy()

        # Create masks (these are correctly (m, n))
        p_constrained_mask = alpha_star_P < L_P
        b_constrained_mask = alpha_star_B < L_B
        
        # --- Case 3: P is constrained ---
        # All arrays in this calculation are now full (m, n) shape, so masking works.
        alpha_b_on_p_boundary = -(g_B_full + H_pb * L_P) / (H_BB_full + 1e-20)
        
        alpha_p_opt[p_constrained_mask] = L_P[p_constrained_mask]
        alpha_b_opt[p_constrained_mask] = np.maximum(
            L_B_full[p_constrained_mask], 
            alpha_b_on_p_boundary[p_constrained_mask]
        )
        
        # --- Case 4: b is constrained ---
        # All arrays in this calculation are also now full (m, n) shape.
        alpha_p_on_b_boundary = -(g_P + H_pb * L_B_full) / H_pp

        # This line, which previously failed, now works because L_B_full is (m, n).
        alpha_b_opt[b_constrained_mask] = L_B_full[b_constrained_mask]
        alpha_p_opt[b_constrained_mask] = np.maximum(
            L_P[b_constrained_mask], 
            alpha_p_on_b_boundary[b_constrained_mask]
        )

        # --- Case 2: Both are constrained (overwrites the above where true) ---
        both_constrained_mask = p_constrained_mask & b_constrained_mask
        alpha_p_opt[both_constrained_mask] = L_P[both_constrained_mask]
        alpha_b_opt[both_constrained_mask] = L_B_full[both_constrained_mask]
        
        return alpha_p_opt, alpha_b_opt

    def optimize_unified_gcd(self, num_iterations=100, verbose=True):
        """
        Main optimizer using Greedy Coordinate Descent on (P, b) blocks.
        """
        P = self.P.copy()
        b = self.b.copy()

        # --- Pre-computation ---
        print("Pre-computing Hessian components and initial gradients...")
        H_pp = self.mu1 + self.mu2
        H_pb = -self.mu2
        g_norm_squared = np.sum(np.abs(self.G)**2, axis=0)
        H_bb_vec = self.lambda_ * (g_norm_squared**2) + self.mu2

        inv_det_H = 1.0 / (H_pp * H_bb_vec[:, np.newaxis] - H_pb**2)
        
        grad_P = self._compute_gradient_P(P, b)
        grad_b = self._compute_gradient_b(b, P.sum(axis=1))
        
        gtg_cache = {}

        print("Starting optimization...")
        start_time = time.time()
        for it in range(num_iterations):
            # --- 1. Vectorized Search for Best Block Update ---
            alpha_p_opts, alpha_b_opts = self._solve_all_2d_subproblems(
                grad_P, grad_b, P, b, H_pp, H_pb, H_bb_vec, inv_det_H)
            
            # --- 2. Calculate Potential Improvement (Delta) ---
            g_B = grad_b[:, np.newaxis]
            H_BB = H_bb_vec[:, np.newaxis]
            delta = - (grad_P * alpha_p_opts + g_B * alpha_b_opts + \
                       0.5 * (H_pp * alpha_p_opts**2 + 2 * H_pb * alpha_p_opts * alpha_b_opts + H_BB * alpha_b_opts**2))
            
            # --- 3. Find and Apply Best Update ---
            i_star, j_star = np.unravel_index(np.argmax(delta), delta.shape)
            max_delta = delta[i_star, j_star]
            
            if max_delta < 1e-12: # Convergence criterion
                print("Converged due to negligible improvement.")
                break
                
            alpha_p = alpha_p_opts[i_star, j_star]
            alpha_b = alpha_b_opts[i_star, j_star]
            
            P[i_star, j_star] += alpha_p
            b[i_star] += alpha_b

            #support update
            if alpha_b == -b[i_star]:
                self.support[i_star] = False
            else:
                self.support[i_star] = True
            
            # --- 4. Efficient Gradient Updates ---
            grad_P[:, j_star] += self.mu1 * alpha_p
            grad_P[i_star, :] += self.mu2 * (alpha_p - alpha_b)

            # Check if we have computed this column before.
            if i_star in gtg_cache:
                # Cache HIT: Retrieve the pre-computed column.
                gtg_col_i_star = gtg_cache[i_star]
            else:
                # Cache MISS: Perform the expensive computation...
                g_i_star = self.G[:, i_star]
                gtg_col_i_star = self.G.conj().T @ g_i_star
                # ...and store the result in the cache for next time.
                gtg_cache[i_star] = gtg_col_i_star

            cmf_grad_update = self.lambda_ * alpha_b * np.real(np.abs(gtg_col_i_star)**2)
            grad_b += cmf_grad_update
            grad_b[i_star] -= self.mu2 * (alpha_p - alpha_b)
            
            if verbose and (it % 100 == 0 or it == num_iterations - 1):
                print(f"[Iter {it:04d}] Loss: {self.loss_function_split():.8f}, delta={max_delta:.3e}, "
                      f"Updating (P({i_star},{j_star}), b({i_star}))")

        self.P = P
        self.b = b
        duration = time.time() - start_time
        print("Unified GCD Optimization finished!")
        print(f"Total optimization time: {duration:.2f} seconds")

    def get_detections(self,xg,yg,zg):
        P_fusion=self.P
        detections=[]

        for i in range(P_fusion.shape[1]):
                    b_source = P_fusion[:,i]
                    b_source_reshaped = b_source.reshape(len(yg), len(xg), len(zg))
                    max_indices_3d = np.unravel_index(np.argmax(b_source_reshaped), b_source_reshaped.shape)

                    # The indices correspond to the positions in your xg, yg, and zg arrays
                    estimated_x_index = max_indices_3d[1]
                    estimated_y_index = max_indices_3d[0]
                    estimated_z_index = max_indices_3d[2]

                    # Get the corresponding coordinates from your grid arrays
                    estimated_x_coord = xg[estimated_x_index]
                    estimated_y_coord = yg[estimated_y_index]
                    estimated_z_coord = zg[estimated_z_index]

                    #get their power (hopefully 1/N)
                    Ppower=b_source_reshaped[estimated_y_index,estimated_x_index,estimated_z_index]

                    #get their acoustic power from b 
                    bpower=self.b[estimated_y_index*len(xg)*len(zg)+estimated_x_index*len(zg)+estimated_z_index]

                    detections.append([estimated_x_coord, estimated_y_coord, estimated_z_coord,Ppower,bpower])
        return detections