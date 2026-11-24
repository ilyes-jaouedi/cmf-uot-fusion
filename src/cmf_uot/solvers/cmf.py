import numpy as np
import time

# the classical CMF like the acoustic one argmin||G diag(p)G^H - sigma||_F^2

class CMFSolver:
    def __init__(self,G,sigma):
        self.G=G
        self.sigma=sigma
        self.M, self.m = G.shape 
        self.p = np.zeros(self.m)
        self.support = np.zeros(self.m, dtype=bool)

    def loss(self, p):
         R = self.G @ (p * self.G.conj()).T - self.sigma 
         return 0.5 * np.linalg.norm(R, ord='fro') ** 2
    
    def compute_gradient(self):
        R = self.G @ (self.p * self.G.conj()).T - self.sigma  # (M, M)
        GR = R @ self.G                                       # (M, m)
        gRu = np.einsum('mc,mc->c', self.G.conj(), GR).real   # (m,)
        return gRu
    
    def GCD(self,nb_iterations=1000,verbose=False,inside=True):
        start_time = time.time()
        p = self.p.copy()  # shape: (m, n)
        
        g_norm_squared = np.sum(np.abs(self.G)**2, axis=0)           # shape: (m,)
        grad = self.compute_gradient()                              # shape: (m, n)
        abs_squared_set = {}

        for it in range(nb_iterations + 1):
            # Compute Lipschitz matrix L_ij
            L = g_norm_squared**2   

            # Compute optimal step size α*
            alpha_star = -grad / L                                   # shape: (m, n)

            # Project to non-negativity constraint
            alpha = np.maximum(-p, alpha_star)                       # shape: (m, n)

            # Compute predicted decrease (delta)
            delta = -alpha * grad + 0.5 * L * (alpha**2)             # shape: (m, n)

            # Select best i
            best_i=np.argmax(delta)
            
            best_alpha = alpha[best_i]

            if best_alpha == -p[best_i]:
                self.support[best_i] = False
            else:
                self.support[best_i] = True

            max_delta = delta[best_i]

            # Apply update to P
            p[best_i] += best_alpha
            self.P = p

            # Cache or retrieve |A[:, i]|² = |Gᴴ G[:, i]|²
            if best_i not in abs_squared_set:
                dot_products = self.G.conj().T @ self.G[:, best_i]         # shape: (m,)
                abs_squared = np.abs(dot_products)**2                      # shape: (m,)
                abs_squared_set[best_i] = abs_squared
            else:
                abs_squared = abs_squared_set[best_i]

            # Update gradient (vectorized)
            grad += best_alpha * abs_squared

            if verbose:
                print(f"[Iter {it}] Loss: {self.loss():.8f}, delta = {max_delta:.3e}, P[{best_i}] += {best_alpha:.8e}")

        self.p = p
        duration = time.time() - start_time
        if inside==False:
            print("Greedy Coordinate Descent Optimization finished!")
            # Summary of A-usage
            percent_used = 100.0 * len(abs_squared_set) / self.m
            print(f"[G-usage] Used {len(abs_squared_set)} / {self.m} columns of G ({percent_used:.2f}%) during optimization.")
            print(f"Total optimization time: {duration:.2f} seconds")

