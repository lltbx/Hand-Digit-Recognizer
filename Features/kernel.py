import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    dot_products = X @ Y.T
    
    # Add constant c and raise to power p
    kernel_matrix = (dot_products + c) ** p
    
    return kernel_matrix



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)  # (n, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)  # (1, m)
    
    # Compute squared Euclidean distances: ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 x_i . y_j
    distances = X_norm + Y_norm - 2 * (X @ Y.T)  # (n, m)
    
    # Compute the kernel: exp(-gamma * ||x-y||^2)
    kernel_matrix = np.exp(-gamma * distances)
    
    return kernel_matrix
