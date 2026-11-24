import numpy as np
import time

def est_lambda(Sigma,nb_sources,Nseg):
    eigenvalues = np.linalg.eigh(Sigma)[0]
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    noise_eigenvalues = eigenvalues_sorted[nb_sources:]
    sigma_bruit_squared = np.mean(noise_eigenvalues)
    lambda_prime = Nseg / sigma_bruit_squared
    return lambda_prime

class TransportOptimizer_numpy:
    def __init__(self, C,a,sigma,G, lambda_=1.0, mu=1.0):
        self.m = C.shape[0] #num_grid_points
        self.n = C.shape[1] #num_camera_detections
        self.lambda_ = lambda_
        self.mu = mu
        self.P = np.zeros((self.m, self.n))
        self.C = C#/np.linalg.norm(C,ord=2)
        self.G = G
        sigma = sigma - np.diag(np.diag(sigma))
        self.sigma = sigma
        self.a = a
        self.support = np.zeros(self.m, dtype=bool)

    def loss_function(self):
        term1 = np.sum(self.C * self.P)

        # CMF term: G diag(p) Gᴴ - sigma
        R = self.G @ (self.P.sum(axis=1) * self.G.conj()).T - self.sigma
        term2 = (self.lambda_ / 2) * np.linalg.norm(R, ord='fro') ** 2

        # Marginal constraint term
        residual = self.P.sum(axis=0) - self.a
        term3 = (self.mu / 2) * np.linalg.norm(residual, ord=2) ** 2

        return term1 + term2 + term3
    
    def compute_gradient(self,P):
        """
        Compute the full gradient ∇𝓛(P).

        Args:
            P: (m, n) real-valued transport matrix
            self.C: (m, n) camera cost matrix
            self.D: (M, m) complex steering matrix G
            self.sigma: (M, M) complex Hermitian matrix (covariance)
            self.a: (n,) target column marginal
            self.lambda_, self.mu: scalar weights

        Returns:
            grad: (m, n) array representing ∇𝓛(P)
        """
        col_sums = P.sum(axis=0) # (n,)

        # Compute R = G diag(p) Gᴴ - Sigma
        R = self.G @ (P.sum(axis=1) * self.G.conj()).T - self.sigma      # (M, M)

        # Compute all Re(g_uᴴ R g_u) for u = 0 to m-1
        # Efficiently: diag(Gᴴ R G)
        GR = R @ self.G                                 # (M, m)
        gRu = np.einsum('mc,mc->c', self.G.conj(), GR).real  # (m,)

        # Broadcast to (m, n)
        Ru_term = self.lambda_ * gRu[:, np.newaxis]     # (m, 1) → (m, n)
        mu_term = self.mu * (col_sums[np.newaxis, :] - self.a[np.newaxis, :])  # (1, n) → (m, n)

        grad = self.C + Ru_term + mu_term
        return grad
    
    def optimize_greedy_coordinate_descent(self, num_iterations=100, verbose=True):
        """
        Greedy coordinate descent using Gauss-Southwell rule (NumPy version).
        Minimizes: ⟨C, P⟩ + λ/2 ||G diag(P1) G^H - Sigma||_F^2 + μ/2 ||Pᵗ1 - a||²
        """
        start_time = time.time()
        P = self.P.copy()  # shape: (m, n)

        g_norm_squared = np.sum(np.abs(self.G)**2, axis=0)           # shape: (m,)
        grad = self.compute_gradient(P)                              # shape: (m, n)
        abs_squared_set = {}

        for it in range(num_iterations + 1):
            # Compute Lipschitz matrix L_ij
            L = self.lambda_ * g_norm_squared[:, None]**2 + self.mu  # shape: (m, n)

            # Compute optimal step size α*
            alpha_star = -grad / L                                   # shape: (m, n)

            # Project to non-negativity constraint
            alpha = np.maximum(-P, alpha_star)                       # shape: (m, n)

            # Compute predicted decrease (delta)
            delta = -alpha * grad + 0.5 * L * (alpha**2)             # shape: (m, n)

            # Select best (i, j)
            max_idx = np.unravel_index(np.argmax(delta), delta.shape)
            best_i, best_j = max_idx
            best_alpha = alpha[best_i, best_j]

            if best_alpha == -P[best_i, best_j]:
                self.support[best_i] = False
            else:
                self.support[best_i] = True
                
            max_delta = delta[best_i, best_j]

            # Apply update to P
            P[best_i, best_j] += best_alpha
            self.P = P

            # Cache or retrieve |A[:, i]|² = |Gᴴ G[:, i]|²
            if best_i not in abs_squared_set:
                dot_products = self.G.conj().T @ self.G[:, best_i]         # shape: (m,)
                abs_squared = np.abs(dot_products)**2                      # shape: (m,)
                abs_squared_set[best_i] = abs_squared
            else:
                abs_squared = abs_squared_set[best_i]

            # Update gradient (vectorized)
            grad += best_alpha * self.lambda_ * abs_squared[:, None]
            grad[:, best_j] += best_alpha * self.mu

            if verbose:
                print(f"[Iter {it}] Loss: {self.loss_function():.8f}, delta = {max_delta:.3e}, P[{best_i},{best_j}] += {best_alpha:.8e}")

        self.P = P
        duration = time.time() - start_time
        print("Greedy Coordinate Descent Optimization finished!")
        # Summary of A-usage
        percent_used = 100.0 * len(abs_squared_set) / self.m
        print(f"[G-usage] Used {len(abs_squared_set)} / {self.m} columns of G ({percent_used:.2f}%) during optimization.")
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
                    detections.append([estimated_x_coord, estimated_y_coord, estimated_z_coord])
        return detections