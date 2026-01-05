
import numpy as np
from itertools import combinations
import pyomo.environ as pyo

class QuadraticSurrogate:
    """
    A class to manage a quadratic surrogate model Q(x) = c + b^T x + 0.5 * x^T H x.
    
    Attributes:
        dim (int): Dimension of the input space.
        X (list): List of sample points (numpy arrays).
        y (list): List of sample values (floats).
        c (float): Constant term.
        b (np.array): Linear coefficients (shape: dim).
        H (np.array): Hessian matrix (shape: dim x dim, symmetric).
        fitted (bool): Whether the model has been fitted.
    """
    def __init__(self, dim, ridge=1e-8):
        self.dim = dim
        self.ridge = ridge
        self.X = []
        self.y = []
        
        # Model parameters
        self.c = 0.0
        self.b = np.zeros(dim)
        self.H = np.zeros((dim, dim))
        self.fitted = False

    def add_point(self, x, val):
        """Adds a sample point (x, val) to the dataset."""
        if len(x) != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {len(x)}")
        self.X.append(np.array(x))
        self.y.append(val)
        self.fitted = False

    def fit(self):
        """Fits the quadratic model using Ridge regression."""
        print(f"  [Q-Surrogate] Fit called. N points: {len(self.X)}")
        if not self.X:
            return

        N = len(self.X)
        X_arr = np.array(self.X) # N x dim
        y_arr = np.array(self.y) # N
        
        # Construct feature matrix for [c, b_1, ..., b_d, H_11, H_12, ..., H_dd]
        # Number of features: 1 (c) + d (b) + d(d+1)/2 (H upper triangular)
        num_features = 1 + self.dim + (self.dim * (self.dim + 1)) // 2
        
        if N < num_features:
            # Not enough points to fit fully
            pass 

        print(f"  [Q-Surrogate] Constructing Phi ({N} x {num_features})...")
        Phi = np.zeros((N, num_features))
        
        for i in range(N):
            x = X_arr[i]
            col = 0
            
            # Constant term
            Phi[i, col] = 1.0; col += 1
            
            # Linear terms
            for j in range(self.dim):
                Phi[i, col] = x[j]; col += 1
            
            # Quadratic terms
            for j in range(self.dim):
                for k in range(j, self.dim):
                    if j == k:
                        Phi[i, col] = 0.5 * x[j]**2
                    else:
                        Phi[i, col] = x[j] * x[k]
                    col += 1
                    
        # Solve (Phi^T Phi + lambda I) w = Phi^T y
        print("  [Q-Surrogate] Solving linear system (Pure Python)...")
        # Solve (Phi^T Phi + lambda I) w = Phi^T y
        print("  [Q-Surrogate] Solving linear system (Pure Python)...")
        # reg_matrix = self.ridge * np.eye(num_features) # Unsafe if numpy is broken
        
        try:
            # LHS = Phi.T @ Phi + reg_matrix
            # RHS = Phi.T @ y_arr
            
            # --- PURE PYTHON MATRIX MULTIPLICATION START ---
            print("  [Q-Surrogate] Computing LHS/RHS in pure Python...")
            Phi_list = Phi.tolist()
            y_list = y_arr.tolist()
            
            # LHS is num_features x num_features
            LHS = [[0.0] * num_features for _ in range(num_features)]
            
            # RHS is num_features
            RHS = [0.0] * num_features
            
            # Compute Phi.T @ Phi and Phi.T @ y
            # This is O(N * num_features^2), which is fine for small N, d
            for i in range(num_features):
                # Compute RHS[i] = column i of Phi dot y
                val_rhs = 0.0
                for k in range(N):
                    val_rhs += Phi_list[k][i] * y_list[k]
                RHS[i] = val_rhs
                
                for j in range(i, num_features): # Symmetric
                    val_lhs = 0.0
                    for k in range(N):
                        val_lhs += Phi_list[k][i] * Phi_list[k][j]
                    LHS[i][j] = val_lhs
                    LHS[j][i] = val_lhs
            
            # Add Ridge
            for i in range(num_features):
                LHS[i][i] += self.ridge
            
            print(f"  [Q-Surrogate] LHS/RHS computed. Size: {len(LHS)}")
            # --- PURE PYTHON MATRIX MULTIPLICATION END ---
            
            
            # --- PURE PYTHON SOLVER START ---
            # Using Gaussian Elimination to avoid MKL/BLAS hard crashes
            A = LHS # Already a list of lists
            b = RHS # Already a list
            n = len(b)
            
            # Augment
            for i in range(n):
                A[i].append(b[i])
            
            # Forward elimination
            for i in range(n):
                # Pivot
                max_el = abs(A[i][i])
                max_row = i
                for k in range(i+1, n):
                    if abs(A[k][i]) > max_el:
                        max_el = abs(A[k][i])
                        max_row = k
                
                # Swap
                A[i], A[max_row] = A[max_row], A[i]
                
                # Make 0
                for k in range(i+1, n):
                    c = -A[k][i] / (A[i][i] + 1e-12) # epsilon
                    for j in range(i, n+1):
                        if i == j:
                            A[k][j] = 0
                        else:
                            A[k][j] += c * A[i][j]
            
            # Back substitution
            x = [0.0] * n
            for i in range(n-1, -1, -1):
                x[i] = A[i][n] / (A[i][i] + 1e-12)
                for k in range(i-1, -1, -1):
                    A[k][n] -= A[k][i] * x[i]
            
            w = np.array(x)
            # --- PURE PYTHON SOLVER END ---

        except Exception as e:
            print(f"  [Q-Surrogate] Error in solve: {e}")
            self.fitted = False
            return

        # Unpack weights
        try:
            print("  [Q-Surrogate] Unpacking weights...")
            col = 0
            self.c = w[col]; col += 1
            
            self.b = np.zeros(self.dim)
            for j in range(self.dim):
                self.b[j] = w[col]; col += 1
                
            self.H = np.zeros((self.dim, self.dim))
            for j in range(self.dim):
                for k in range(j, self.dim):
                    val = w[col]
                    self.H[j, k] = val
                    self.H[k, j] = val # Symmetric
                    col += 1
            
            self.fitted = True
            print("  [Q-Surrogate] Fit complete.")
        except Exception as e:
            print(f"Error unpacking: {e}")
            self.fitted = False

    def evaluate(self, x):
        """Evaluates Q(x)."""
        # x = np.array(x)
        # return self.c + np.dot(self.b, x) + 0.5 * x.T @ self.H @ x
        
        # Pure Python implementation
        val = self.c
        
        # Linear term: b @ x
        for j in range(self.dim):
            val += self.b[j] * x[j]
            
        # Quadratic term: 0.5 * x.T @ H @ x
        # 0.5 * sum(x_i * H_ij * x_j)
        quad_term = 0.0
        for i in range(self.dim):
            row_val = 0.0
            for j in range(self.dim):
                row_val += self.H[i, j] * x[j]
            quad_term += x[i] * row_val
            
        val += 0.5 * quad_term
        return val

    def build_pyomo_expression(self, vars_list):
        """
        Builds a Pyomo expression for Q(x).
        vars_list: list of Pyomo variables corresponding to x dimensions in order.
        """
        if not self.fitted:
            # Return 0 if not fitted? Or error?
            # To be safe for "optional" behavior, return 0.0
            return 0.0
        
        expr = self.c
        
        # Linear
        for j, var in enumerate(vars_list):
            expr += self.b[j] * var
            
        # Quadratic
        # 0.5 * sum(x_j * H_jk * x_k)
        # To avoid double counting and ensure correctness:
        # sum_{j} 0.5 * H_jj * x_j^2 + sum_{j<k} H_jk * x_j * x_k
        
        for j in range(self.dim):
            for k in range(j, self.dim):
                if j == k:
                    expr += 0.5 * self.H[j, j] * (vars_list[j]**2)
                else:
                    # Off-diagonal appears twice in 0.5*x.T*H*x (H_jk and H_kj), so 2 * 0.5 * H_jk * xj * xk = H_jk * xj * xk
                    expr += self.H[j, k] * vars_list[j] * vars_list[k]
                    
        return expr

    def min_over_box(self, lbs, ubs):
        """
        Computes the exact global minimum of Q(x) over the box [lbs, ubs].
        Uses active set / face enumeration.
        Only feasible for small dimensions (e.g., dim <= 6).
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        
        dim = self.dim
        min_val = float('inf')
        
        # Check interior stationary point first?
        # Actually easier to just iterate all 3^d faces (fix some vars to L, some to U, some free)
        # But for d=3, 3^3=27, trivial. For d=6, 729, ok. 
        
        # Enumerate all faces. A face is defined by partition of indices into (Lower, Upper, Free)
        # For each index i in range(dim): state[i] \in {-1 (Lower), 0 (Free), 1 (Upper)}
        
    def min_over_box(self, lbs, ubs):
        """
        Approximates the minimum of Q(x) over the box [lbs, ubs] via random sampling.
        Exact minimization is replaced by sampling to avoid BLAS crashes.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        
        dim = self.dim
        min_val = float('inf')
        
        import random
        
        # Corner enumeration (2^d) - critical for convex/concave
        import itertools
        corners = list(itertools.product(*zip(lbs, ubs)))
        for pt in corners:
             val = self.evaluate(pt)
             if val < min_val:
                 min_val = val

        # Random sampling interior
        # 100 points
        for _ in range(100):
            pt = []
            for i in range(dim):
                pt.append(random.uniform(lbs[i], ubs[i]))
            
            val = self.evaluate(pt)
            if val < min_val:
                min_val = val
                
        return min_val
                
        return min_val
