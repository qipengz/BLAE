import numpy as np
import torch

from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.decomposition import PCA



def get_Distance_Mat(X, n_neighbors=None, radius=None, Geodesic=True):
    if Geodesic:
        embedding = Isomap(n_neighbors=n_neighbors, radius=radius, n_components=2)
        embedding.fit(X.reshape(X.shape[0], -1))
        D = embedding.dist_matrix_
        if n_neighbors:
            dist_list, indices = embedding.nbrs_.kneighbors(X, n_neighbors, return_distance=True)
            return D, dist_list[:, 1:], indices[:, 1:]
        elif radius:
            dist_list, indices = embedding.nbrs_.radius_neighbors(X, radius, return_distance=True)
            return D, dist_list, indices
    else:
        # Compute the pairwise distance matrix
        X = X.reshape(X.shape[0], -1)
        D = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(i + 1, X.shape[0]):
                D[i, j] = np.linalg.norm(X[i] - X[j])
                D[j, i] = D[i, j]
        return D, None, None


def compute_tangent_space(X, indices):
    # use pca to compute the tangent space at each point
    pca = PCA(n_components=2)
    tangent_spaces = []
    for i in range(X.shape[0]):
        # get the neighbors of the point
        neighbors = X[indices[i]]
        # fit pca to the neighbors
        pca.fit(neighbors)
        # get the tangent space
        tangent_space = pca.components_
        tangent_spaces.append(tangent_space.T)
    return np.array(tangent_spaces)
    

def compute_jacobian(model, x):
    """
    compute the jacobian of a neural network model with respect to the input x
    
    input:
    model: instance of nn.Module
    x: torch.tensor [batch_size, input_dim]
    
    return:
    jacobian: [batch_size, output_dim, input_dim]
    """
    x.requires_grad_(True)
    y = model(x)
    y = y.view(y.size(0), -1)
    
    batch_size = x.size(0)
    input_dim = x.size(1)
    output_dim = y.size(1)
    
    # initialize jacobian
    jacobian = torch.zeros(batch_size, output_dim, input_dim, device=x.device)
    
    # compute jacobian
    for i in range(output_dim):
        # zero gradients
        grad_output = torch.zeros_like(y)
        grad_output[:, i] = 1.0
        
        # compute gradient
        grad = torch.autograd.grad(y, x, grad_outputs=grad_output, 
                                   create_graph=True,
                                   #is_grads_batched=True,
                                   retain_graph=True)[0]
        
        # store gradient
        jacobian[:, i, :] = grad

    return jacobian
                
                
def compute_euclidean_knn_distance_matrix(X, k=10, limit=1e4):
    """
    Computes a geodesic distance matrix using K-nearest neighbors + shortest paths,
    automatically handling both 2D data (N, d) and 4D RGB data (N, C, H, W).

    Args:
        X (torch.Tensor or np.ndarray): 
            - shape (N, d), or
            - shape (N, C, H, W) for image data
        k (int): number of neighbors for the KNN graph
        limit (float): maximum path length used by Dijkstra

    Returns:
        dist_mat (torch.Tensor): shape (N, N),
            where dist_mat[i,j] is the geodesic distance from i to j
            under a KNN graph.

    Example:
        # If 'X' is shape (N, 3, H, W), we flatten to (N, 3*H*W)
        dist_mat = compute_euclidean_knn_distance_matrix(X, k=10, limit=1e4)
    """
    
    # 1) Convert torch.Tensor -> numpy, flatten if needed
    if isinstance(X, torch.Tensor):
        # Ensure on CPU
        X = X.cpu()

        # If shape is (N, C, H, W), flatten to (N, C*H*W)
        if X.dim() == 4:
            N, C, H, W = X.shape
            X = X.view(N, C * H * W)  # => (N, d)

        elif X.dim() == 2:
            # shape (N, d), do nothing
            pass
        else:
            raise ValueError(
                f"compute_euclidean_knn_distance_matrix expects shape (N, d) or (N, C, H, W), "
                f"but got {tuple(X.shape)}"
            )

        X = X.numpy()  # final shape => (N, d)

    # 2) If already a NumPy array, flatten if needed
    elif isinstance(X, np.ndarray):
        if X.ndim == 4:
            # shape => (N, C, H, W)
            N, C, H, W = X.shape
            X = X.reshape(N, C * H * W)  # => (N, d)
        elif X.ndim == 2:
            # shape => (N, d), do nothing
            pass
        else:
            raise ValueError(
                f"compute_euclidean_knn_distance_matrix expects shape (N, d) or (N, C, H, W), "
                f"but got {X.shape}"
            )
    else:
        raise TypeError("X must be either a torch.Tensor or a np.ndarray.")

    # 3) Build the KNN graph (using Euclidean distances)
    G = kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    # G = radius_neighbors_graph(X, radius=4.5, mode='distance', include_self=False)
    graph = csr_matrix(G)

    # 4) Run Dijkstra to find geodesic distances
    dist_mat_np = dijkstra(csgraph=graph, directed=False, limit=limit)
    # Replace invalid values with 'limit'
    dist_mat_np = np.nan_to_num(dist_mat_np, nan=limit, posinf=limit, neginf=0.0)

    # 5) Convert back to torch
    dist_mat = torch.tensor(dist_mat_np, dtype=torch.float32)
    return dist_mat


def get_laplacian(
    X, 
    distfunc_name='Euclidean_knn', 
    bandwidth=1.0, 
    precomputed_dist=None, 
    k=5, 
    limit=1e4
):
    """
    Build a Laplacian from data X (shape: (B, N, d)) using the specified distance function.
    If precomputed_dist is given (shape: (N, N) or (B, N, N)), we use it directly.
    """
    
    B, N, *d = X.shape
    c = 1 / 4

    # If we have a precomputed distance matrix, broadcast/expand as needed
    if precomputed_dist is not None:
        if precomputed_dist.dim() == 2:
            # shape (N, N) -> expand to (B, N, N)
            dist_XX = precomputed_dist.unsqueeze(0).expand(B, -1, -1).to(X.device)
        else:
            # assume shape (B, N, N)
            dist_XX = precomputed_dist.to(X.device)
    else:
        # If not provided, compute from scratch (Euclidean or KNN).
        if distfunc_name == 'Euclidean':
            X_ = X.unsqueeze(2)  # (B, N, 1, d)
            X2_ = X.unsqueeze(1)  # (B, 1, N, d)
            dist_XX = (X_ - X2_).pow(2).sum(dim=-1).sqrt()  # (B, N, N)
        elif distfunc_name == 'Euclidean_knn':
            dist_list = []
            X_np = X.cpu().numpy()  # shape (B, N, d)
            for single_batch in X_np:
                G = kneighbors_graph(single_batch, n_neighbors=k, mode='distance', include_self=False)
                graph = csr_matrix(G)
                dist_mat_np = dijkstra(csgraph=graph, directed=False, limit=limit)
                dist_mat = torch.tensor(dist_mat_np, dtype=X.dtype)
                dist_mat = dist_mat.nan_to_num(posinf=limit)
                dist_list.append(dist_mat)
            dist_XX = torch.stack(dist_list, dim=0)
        else:
            raise NotImplementedError(f"distfunc {distfunc_name} not recognized.")

    # Build the Gaussian kernel
    K = torch.exp(-(dist_XX ** 2) / bandwidth)
    d_i = K.sum(dim=2)              # shape (B, N)
    D_inv = torch.diag_embed(1.0 / d_i)     # (B, N, N)
    K_tilde = D_inv @ K @ D_inv           # (B, N, N)
    sum_over_j = K_tilde.sum(dim=2)       # (B, N)
    D_tilde_inv = torch.diag_embed(1.0 / sum_over_j)
    I_b = torch.diag_embed(torch.ones(B, N, device=X.device))
    L = (D_tilde_inv @ K_tilde - I_b) / (c * bandwidth)  # (B, N, N)
    return L


def get_JGinvJT(L, Z):
    """
    L: (B, N, N)
    Z: (B, N, latent_dim)
    Compute the H_tilde from the original iso-loss approach.
    """
    B, N, n = Z.shape
    
    catY1 = Z.unsqueeze(-1).expand(-1, -1, -1, n)  # (B, N, n, n)
    catY2 = Z.unsqueeze(-2).expand(-1, -1, n, -1)  # (B, N, n, n)
    # Flatten last two dims for matrix multiply
    term1 = (L @ (catY1 * catY2).reshape(B, N, n * n)).reshape(B, N, n, n)

    LY = L @ Z  # (B, N, n)
    LY_j = LY.unsqueeze(-2).expand(-1, -1, n, -1)  # (B, N, n, n)
    LY_i = LY.unsqueeze(-1).expand(-1, -1, -1, n)  # (B, N, n, n)

    term2 = catY1 * LY_j
    term3 = catY2 * LY_i
    H_tilde = 0.5 * (term1 - term2 - term3)  # (B, N, n, n)
    return H_tilde


def relaxed_distortion_measure_JGinvJT(H):
    """
    Distortion measure from H = JGinvJT:
    Typically: Tr(H^2) - 2Tr(H).
    """
    TrH = H.diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)     # shape (B, N)
    H2 = H @ H
    TrH2 = H2.diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)    # shape (B, N)
    return TrH2.mean() - 2.0 * TrH.mean()  # mean over B, N


def relaxed_distortion_measure(func, z, eta=0.2, metric='identity', create_graph=True):
    if metric == 'identity':
        bs = len(z)
        z_perm = z[torch.randperm(bs)]
        if eta is not None:
            alpha = (torch.rand(bs).to(z) * (1 + 2 * eta) - eta).unsqueeze(1).to(z)
            z_augmented = alpha * z + (1 - alpha) * z_perm.to(z)
        else:
            z_augmented = z
        v = torch.randn(z.size()).to(z)
        Jv = torch.autograd.functional.jvp(func, z_augmented, v=v, create_graph=create_graph)[1]
        TrG = torch.sum(Jv.view(bs, -1) ** 2, dim=1).mean()
        JTJv = (torch.autograd.functional.vjp(func, z_augmented, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
        TrG2 = torch.sum(JTJv ** 2, dim=1).mean()
        return TrG2 / TrG ** 2
    else:
        raise NotImplementedError


def get_flattening_scores(G, mode='condition_number'):
    if mode == 'condition_number':
        S = torch.svd(G).S
        scores = S.max(1).values / S.min(1).values
    elif mode == 'variance':
        G_mean = torch.mean(G, dim=0, keepdim=True)
        A = torch.inverse(G_mean) @ G
        scores = torch.sum(torch.log(torch.svd(A).S) ** 2, dim=1)
    else:
        pass
    return scores


def jacobian_decoder_jvp_parallel(func, inputs, v=None, create_graph=True):
    batch_size, z_dim = inputs.size()
    if v is None:
        v = torch.eye(z_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim).to(inputs)
    inputs = inputs.repeat(1, z_dim).view(-1, z_dim)
    jac = (
        torch.autograd.functional.jvp(
            func, inputs, v=v, create_graph=create_graph
        )[1].view(batch_size, z_dim, -1).permute(0, 2, 1)
    )
    return jac


def get_pullbacked_Riemannian_metric(func, z):
    J = jacobian_decoder_jvp_parallel(func, z, v=None)
    G = torch.einsum('nij,nik->njk', J, J)
    return G