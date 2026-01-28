import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from gudhi import RipsComplex
from utils.functionals import relaxed_distortion_measure



class GeodesicBasedLoss(nn.Module):

    def __init__(self):
        super(GeodesicBasedLoss, self).__init__()

    def _get_latent_distance(self, z):
        """Compute the pairwise distance matrix for latent vectors."""
        z = z.view(z.size(0), -1)
        p = torch.cdist(z, z, p=2)
        return p

    def forward(self, x):
        pass


class GeoLoss(GeodesicBasedLoss):

    def __init__(self):
        super(GeoLoss, self).__init__()

    def forward(self, D, z):
        p = self._get_latent_distance(z)
        mask_upper = torch.triu(torch.ones_like(D), diagonal=1).bool()
        p = p[mask_upper]
        D = D[mask_upper]
        loss = torch.mean((p - D) ** 2)
        return loss


class LipLoss(GeodesicBasedLoss):

    def __init__(self):
        super(LipLoss, self).__init__()

    def forward(self, D, z):
        p = self._get_latent_distance(z)
        mask_upper = torch.triu(torch.ones_like(D), diagonal=1).bool()
        mask = D > 0
        lip = p / D
        lip = lip[mask_upper & mask]
        loss = torch.mean((lip - 1) ** 2)
        return loss


class LogLipLoss(GeodesicBasedLoss):

    def __init__(self):
        super(LogLipLoss, self).__init__()

    def forward(self, D, z):
        p = self._get_latent_distance(z)
        mask_upper = torch.triu(torch.ones_like(D), diagonal=1).bool()
        mask = D > 0
        lip = p / D
        lip = lip[mask_upper & mask]
        loss = torch.mean((torch.log(lip)) ** 2)
        return loss


class InjectiveLoss(GeodesicBasedLoss):

    def __init__(self, thresh=0.3):
        super(InjectiveLoss, self).__init__()
        self.__thresh = torch.log(torch.tensor(thresh)).float()

    def forward(self, D, z):
        # if isinstance(D, np.ndarray):
        #     D = torch.tensor(D, dtype=torch.float32)
        p = self._get_latent_distance(z)
        mask_upper = torch.triu(torch.ones_like(D), diagonal=1).bool()
        mask = (D > 0.05 * torch.max(D)) & (p > 0)
        mask = mask & mask_upper
        lip = p[mask] / D[mask]

        loss = torch.mean(1 * F.relu(self.__thresh - torch.log(lip)) + 5 * F.relu(lip - 1.0))
        # loss = torch.mean(1 * F.relu(self.__thresh - torch.log(lip)) + 5 * F.relu(lip - 2.0))
        return loss


class SPAELoss(GeodesicBasedLoss):

    def __init__(self, type='R1'):
        super(SPAELoss, self).__init__()
        self.type = type
        self.epsilon = torch.tensor(1e-12).float()

    def forward(self, D, z):
        D = D.clone()
        p = self._get_latent_distance(z)
        lip = (p + torch.ones_like(p) * self.epsilon.to(p.device)) / (D + torch.ones_like(p) * self.epsilon.to(p.device))

        # remove the diagonal elements
        n = lip.size(0)
        mask = ~torch.eye(n, dtype=bool, device=lip.device)
        lip_no_diag = lip[mask].view(n, n - 1)
        log_lip = torch.log(lip_no_diag)

        if self.type == 'R1':
            # compute variance for each row
            Xi = torch.var(log_lip, dim=1)
            loss = torch.mean(Xi)
        elif self.type == 'R2':
            # compute variance for whole matrix
            loss = torch.var(log_lip)
        elif self.type == 'R3':
            loss = torch.mean(log_lip ** 2)  # compute mean of squared log_lip
        else:
            raise ValueError('Invalid type: %s' % self.type)

        return loss


class TopoLoss(nn.Module):
    """
    Topological loss based on persistent homology using geodesic distances.
    """

    def __init__(self):
        super().__init__()

    def compute_persistence(self, distance_matrix, max_dimension=1):
        """
        Compute persistence diagrams using the Gudhi library.

        Args:
            distance_matrix: Distance matrix to compute persistent homology
            max_dimension: Maximum dimension of simplices

        Returns:
            Tensor of destruction edges
        """
        rc = RipsComplex(distance_matrix=distance_matrix.cpu().numpy(), max_edge_length=np.inf)
        st = rc.create_simplex_tree(max_dimension=max_dimension)
        st.compute_persistence()
        pairs = st.persistence_pairs()
        destroy_edges = [death for birth, death in pairs if len(death) == 2]
        return torch.tensor(destroy_edges, dtype=torch.long)

    def forward(self, D, z):
        """
        Compute the topological loss between input space and latent space.

        Args:
            batch_indices: Indices of batch elements
            Z: Latent representations

        Returns:
            Weighted sum of topological losses
        """
        device = z.device

        # Extract the geodesic distance submatrix
        A_X_geo = D

        # Compute latent space Euclidean distances
        A_Z_euc = torch.cdist(z, z)

        # Compute persistence pairs for both spaces
        pi_X = self.compute_persistence(A_X_geo.cpu(), max_dimension=1)
        pi_Z = self.compute_persistence(A_Z_euc.detach().cpu(), max_dimension=1)

        # Compute bidirectional topological loss
        loss_XZ = 0.5 * torch.norm(A_X_geo[pi_X.to(device)] - A_Z_euc[pi_X.to(device)]) ** 2
        loss_ZX = 0.5 * torch.norm(A_Z_euc[pi_Z.to(device)] - A_X_geo[pi_Z.to(device)]) ** 2

        return loss_XZ + loss_ZX


class IsometricRegularization(nn.Module):
    """
    Isometric regularization.
    """

    def __init__(self, metric='identity'):
        super(IsometricRegularization, self).__init__()
        self.metric = metric

    def forward(self, z, decoder, eta=None):
        iso_loss = relaxed_distortion_measure(decoder, z, eta=eta, metric=self.metric)
        return iso_loss