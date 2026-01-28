import numpy as np
import scipy
import copy
import torch
import torch.nn as nn
import torch.func as func
import torch.nn.functional as F

from scipy.sparse.linalg import eigs, eigsh
from scipy import sparse
import scipy.sparse.linalg as spsl
from scipy.linalg import eigh
import warnings

from networkx import sigma
from sklearn.decomposition import PCA
from utils.datasets import CustomDataset
from pydiffmap import diffusion_map as dm

from torch.utils.data import DataLoader

import utils.regularizations
from utils.functionals import get_laplacian, get_JGinvJT, relaxed_distortion_measure_JGinvJT



class BasicTrainer(nn.Module):

    def __init__(self, optimizer, scheduler=None, device='cpu'):
        super(BasicTrainer, self).__init__()
        self.optim = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = None
        self.test_loader = None

    def get_dataloader(self, X_train, y_train, X_test, y_test, batch_size):
        # Convert data to tensors and move to device
        X_train = torch.tensor(X_train, device=self.device, dtype=torch.float)
        y_train = torch.tensor(y_train, device=self.device, dtype=torch.float)
        X_test = torch.tensor(X_test, device=self.device, dtype=torch.float)
        y_test = torch.tensor(y_test, device=self.device, dtype=torch.float)

        # Create training dataloader
        self.train_loader = DataLoader(
            dataset=CustomDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        # Create testing dataloader
        self.test_loader = DataLoader(
            dataset=CustomDataset(X_test, y_test),
            batch_size=X_test.shape[0],  # Use the entire test set as one batch
            shuffle=False
        )

    def train(self, model, epochs):
        # Method to be implemented by subclasses
        # Example optimizer options (commented out):
        # self.optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.1, patience=1000)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=10000, gamma=0.99)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[10000,50000], gamma=0.1)
        pass




class AutoencoderTrainer(BasicTrainer):

    def __init__(self, optimizer, scheduler=None, device='cpu'):
        super(AutoencoderTrainer, self).__init__(optimizer=optimizer, scheduler=scheduler, device=device)

    def train(self, model, epochs, verbose=True):
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            # Training phase
            for x, _, _ in self.train_loader:
                model.train()
                model.zero_grad()

                x_hat = model(x)
                train_loss = criterion(x_hat, x)
                train_loss.backward()
                self.optim.step()
            self.scheduler.step()

            # Evaluation phase
            model.eval()
            for x, _, _ in self.test_loader:
                x_hat = model(x)
                test_recon_loss = criterion(x_hat, x)

            if verbose:
                print("Epoch: [{}/{}], \tTrain Loss: {:.6f}\n"
                      "\t\t\tTest Reconstruction Loss: {:.6f}\n".format(
                    epoch + 1,
                    epochs,
                    train_loss.data.item(),
                    test_recon_loss.data.item()
                ))




class GeodesicAutoencoderTrainer(BasicTrainer):

    def __init__(self, regularization, GeoD, lam_geo, optimizer, scheduler=None, device='cpu', thresh=0.3):
        super(GeodesicAutoencoderTrainer, self).__init__(optimizer=optimizer, scheduler=scheduler, device=device)
        self.GeoD = torch.as_tensor(GeoD, device=device, dtype=torch.float)
        self.lam_geo = torch.as_tensor(lam_geo, device=device, dtype=torch.float)
        self.thresh = thresh
        # Initialize the appropriate regularization method
        if regularization == 'GeoLoss':
            self.reg = utils.regularizations.GeoLoss()
        elif regularization == 'LipLoss':
            self.reg = utils.regularizations.LipLoss()
        elif regularization == 'SPAELoss':
            self.reg = utils.regularizations.SPAELoss()
        elif regularization == 'AdaLipLoss':
            self.reg = utils.regularizations.InjectiveLoss(thresh=self.thresh)
        elif regularization == 'LogLipLoss':
            self.reg = utils.regularizations.LogLipLoss()
        elif regularization == 'TopoLoss':
            self.reg = utils.regularizations.TopoLoss()
        else:
            raise ValueError("Invalid regularization type")

    def train(self, model, epochs, verbose=True):
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            for x, _, ids in self.train_loader:
                model.train()
                model.zero_grad()

                # Forward pass through the model
                z_hat = model.encoder(x)
                x_hat = model.decoder(z_hat)

                # Calculate losses
                reg_loss = self.reg(self.GeoD[ids, :][:, ids], z_hat)
                recon_loss = criterion(x_hat, x)
                train_loss = recon_loss + self.lam_geo * reg_loss

                # Backward pass and optimization
                train_loss.backward()
                self.optim.step()
            self.scheduler.step()

            # Evaluation phase
            model.eval()
            for x, _, ids in self.test_loader:
                z_hat = model.encoder(x)
                x_hat = model.decoder(z_hat)
                test_recon_loss = criterion(x_hat, x)

            if verbose:
                print("Epoch: [{}/{}], \tTrain Loss: {:.6f}\t Regularization: {:.6f}\n"
                      "\t\t\tTest Reconstruction Loss: {:.6f}\n".format(
                    epoch + 1,
                    epochs,
                    train_loss.data.item(),
                    reg_loss.data.item(),
                    test_recon_loss.data.item()
                ))
                # current_lr = self.optim.param_groups[0]['lr']
                # print(f'Epoch {epoch}, Current LR: {current_lr}')




class GGAETrainer(BasicTrainer):

    def __init__(
        self,
        lam=1e-2,
        bandwidth=0.2,
        distfunc_name='Euclidean_knn',
        k=5,
        limit=1e4,
        global_dist=None,
        optimizer=None,
        scheduler=None,
        device='cpu'
    ):
        """
        GGAE trainer that adds geometry-based regularization (iso-loss).

        Args:
            lam (float): Weight for the iso-loss term.
            bandwidth (float): Kernel bandwidth for Laplacian.
            distfunc_name (str): 'Euclidean' or 'Euclidean_knn'.
            k (int): Number of neighbors for KNN if distfunc_name='Euclidean_knn'.
            limit (float): Dijkstra path limit for KNN distances.
            global_dist (torch.Tensor): Precomputed global distance matrix of shape (N, N).
        """
        super(GGAETrainer, self).__init__(optimizer=optimizer, scheduler=scheduler, device=device)
        self.lam = torch.tensor(lam, device=device, dtype=torch.float)
        self.bandwidth = torch.tensor(bandwidth, device=device, dtype=torch.float)
        self.distfunc_name = distfunc_name
        self.k = k
        self.limit = limit
        # We store the global distance matrix if available
        self.global_dist = global_dist.to(device) if global_dist is not None else None

    def train(self, model, epochs, verbose=True):
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            total_loss_val = 0.0
            for x, _, idx in self.train_loader:
                model.train()
                model.zero_grad()

                # Forward pass
                z_hat = model.encoder(x)
                x_hat = model.decoder(z_hat)
                recon_loss = criterion(x_hat, x)

                # shape => (1, B, d) for X, (1, B, latent_dim) for Z
                X_ = x.unsqueeze(0)
                Z_ = z_hat.unsqueeze(0)

                # If we have a global distance matrix, slice it
                sub_dist = None
                if self.global_dist is not None:
                    # (N, N) -> slice with idx => shape (B, B)
                    sub_dist = self.global_dist[idx, :][:, idx]

                # Build the Laplacian
                L = get_laplacian(
                    X_,
                    distfunc_name=self.distfunc_name,
                    bandwidth=self.bandwidth,
                    precomputed_dist=sub_dist,
                    k=self.k,
                    limit=self.limit
                )
                # Compute the iso-loss
                H_tilde = get_JGinvJT(L, Z_)
                iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde)
                # Combine the losses
                train_total_loss = recon_loss + self.lam * iso_loss
                train_total_loss.backward()
                self.optim.step()

                total_loss_val += train_total_loss.item()

            avg_train_loss = total_loss_val / len(self.train_loader)

            # Step the schedulers using the average train loss
            self.scheduler.step()

            # Evaluate on test set (optional)
            model.eval()
            test_recon_loss_val = 0.0
            with torch.no_grad():
                for x_test, _, _ in self.test_loader:
                    z_test = model.encoder(x_test)
                    x_test_hat = model.decoder(z_test)
                    test_recon_loss_val += criterion(x_test_hat, x_test).item()
            test_recon_loss_val /= len(self.test_loader)

            if verbose:
                print(
                    f"Epoch: [{epoch+1}/{epochs}], "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Iso Loss: {iso_loss.item():.6f}, "
                    f"Test Recon Loss: {test_recon_loss_val:.6f}"
                )




class GradientAutoencoderTrainer(BasicTrainer):

    def __init__(self,
                 GeoD,
                 reg_grad='bi-lipschitz',
                 reg_geo='InjectiveLoss',
                 lam_grad=0,
                 lam_geo=0,
                 L=1,
                 optimizer=None,
                 scheduler=None,
                 device='cpu',
                 return_loss=False,
                 thresh=0.3):
        super(GradientAutoencoderTrainer, self).__init__(optimizer=optimizer, scheduler=scheduler, device=device)

        # Convert GeoD to tensor
        if not isinstance(GeoD, torch.Tensor):
            GeoD = torch.tensor(GeoD, dtype=torch.float)
        self.GeoD = GeoD.to(self.device)

        # Convert L, lam_grad, lam_geo to tensors
        for name, val in [('L', L), ('lam_grad', lam_grad), ('lam_geo', lam_geo)]:
            tensor_val = val if isinstance(val, torch.Tensor) else torch.tensor(val, dtype=torch.float)
            setattr(self, name, tensor_val.to(self.device))

        self.reg_grad = reg_grad
        self.reg_geo_name = reg_geo
        self.return_loss = return_loss
        self.thresh = thresh
        # Initialize regularization object
        if reg_geo == 'GeoLoss':
            self.reg_geo = utils.regularizations.GeoLoss()
        elif reg_geo == 'LipLoss':
            self.reg_geo = utils.regularizations.LipLoss()
        elif reg_geo == 'SPAELoss':
            self.reg_geo = utils.regularizations.SPAELoss()
        elif reg_geo == 'InjectiveLoss':
            self.reg_geo = utils.regularizations.InjectiveLoss(thresh=self.thresh)
        elif reg_geo == 'LogLipLoss':
            self.reg_geo = utils.regularizations.LogLipLoss()
        else:
            raise ValueError("Invalid regularization type")

    def train(self, model, epochs, verbose=True):
        criterion = nn.MSELoss()
        if self.return_loss:
            mse_loss_list, geo_loss_list, grad_loss_list = [], [], []

        for epoch in range(epochs):
            for x, _, ids in self.train_loader:
                model.train()
                model.zero_grad()
                z_hat = model.encoder(x)
                x_hat = model.decoder(z_hat)

                # Randomly select mini batch from z_hat
                batch_size = z_hat.size(0)
                # indices = torch.randperm(batch_size)[:round(batch_size * 1)]
                indices = torch.randperm(batch_size)[:round(batch_size * 0.3)]
                z_selected = z_hat[indices]

                # Reconstruction loss & geodesic loss
                recon_loss = criterion(x_hat, x)
                geo_loss = self.reg_geo(self.GeoD[ids, :][:, ids], z_hat)

                # Calculate gradient loss based on regularization type
                if self.reg_grad == 'isometric':
                    jac_dec = func.vmap(func.jacfwd(model.decoder))(z_selected)
                    jac_dec = jac_dec.view(jac_dec.size(0), -1, jac_dec.size(-1))
                    riem_metric_dec = torch.bmm(jac_dec.transpose(1, 2), jac_dec)
                    grad_loss = torch.mean(torch.norm(
                        riem_metric_dec - torch.eye(z_hat.shape[1]).repeat(z_selected.size(0), 1, 1),
                        p='fro', dim=(1, 2)) ** 2)
                    train_loss = recon_loss + self.lam_grad * grad_loss + self.lam_geo * geo_loss

                elif self.reg_grad == 'isometric_enc':
                    jac_enc = func.vmap(func.jacrev(model.encoder))(x[indices])
                    jac_enc = jac_enc.view(jac_enc.size(0), -1, jac_enc.size(-1))
                    riem_metric_enc = torch.bmm(jac_enc.transpose(1, 2), jac_enc)
                    A = torch.bmm(
                        riem_metric_enc - torch.eye(x_hat.shape[1]).repeat(z_selected.size(0), 1, 1),
                        self.tangent_spaces[ids])
                    grad_loss = torch.mean(torch.norm(A, p='fro', dim=(1, 2)) ** 2)
                    train_loss = recon_loss + self.lam_grad * grad_loss + self.lam_geo * geo_loss

                elif self.reg_grad == 'contractive':
                    jac_enc = func.vmap(func.jacrev(model.encoder))(x[indices])
                    jac_enc = jac_enc.reshape(jac_enc.size(0), -1, jac_enc.size(-1))
                    grad_loss = torch.mean(torch.norm(jac_enc, p='fro', dim=(1, 2)) ** 2)
                    train_loss = recon_loss + self.lam_grad * grad_loss + self.lam_geo * geo_loss

                elif self.reg_grad == 'bi-lipschitz':
                    jac_dec = func.vmap(func.jacfwd(model.decoder))(z_selected)
                    jac_dec = jac_dec.view(jac_dec.size(0), -1, jac_dec.size(-1))
                    riem_metric_dec = torch.bmm(jac_dec.transpose(1, 2), jac_dec)

                    diag = torch.diagonal(riem_metric_dec, dim1=1, dim2=2)
                    mask = ~torch.eye(z_hat.shape[1], dtype=bool, device=riem_metric_dec.device)  # [dim, dim]
                    non_diag = riem_metric_dec[:, mask].view(riem_metric_dec.size(0), -1)
                    non_diag_loss = torch.mean(torch.norm(non_diag, p=2, dim=1) ** 2)

                    # Log maybe better?
                    # diag_loss = torch.mean(torch.norm(F.relu(diag - self.L) + 1 * F.relu(1 / self.L - diag), dim=1) ** 2)
                    diag_loss = torch.mean(torch.norm(F.relu(diag - self.L) + 0 * F.relu(1 / self.L - diag), dim=1) ** 2)
                    grad_loss = non_diag_loss + diag_loss
                    train_loss = recon_loss + self.lam_grad * grad_loss + self.lam_geo * geo_loss

                elif self.reg_grad == 'bi-lipschitz_enc+dec':
                    jac_dec = func.vmap(func.jacfwd(model.decoder))(z_selected)
                    jac_dec = jac_dec.view(jac_dec.size(0), -1, jac_dec.size(-1))
                    jac_enc = func.vmap(func.jacrev(model.encoder))(x[indices])
                    jac_enc = jac_enc.view(jac_enc.size(0), -1, jac_enc.size(-1))
                    jac_enc = torch.bmm(jac_enc, self.tangent_spaces[ids])

                    norm_enc = torch.mean(F.relu(torch.linalg.matrix_norm(jac_enc, ord=2) - torch.ones(z_selected.shape[0]) * self.L) ** 2)
                    norm_dec = torch.mean(F.relu(torch.linalg.matrix_norm(jac_dec, ord=2) - torch.ones(z_selected.shape[0]) * self.L) ** 2)
                    
                    grad_loss = norm_enc + norm_dec
                    train_loss = recon_loss + self.lam_grad * grad_loss + self.lam_geo * geo_loss

                elif self.reg_grad == 'bi-lipschitz_dec':
                    jac_dec = func.vmap(func.jacfwd(model.decoder))(z_selected)
                    jac_dec = jac_dec.view(jac_dec.size(0), -1, jac_dec.size(-1))
                    sigma_max = torch.linalg.matrix_norm(jac_dec, ord=2)
                    sigma_min = torch.linalg.matrix_norm(jac_dec, ord=-2)

                    # norm_up = torch.mean(F.relu(torch.log(sigma_max)-torch.log(torch.ones(z_selected.shape[0])*self.L))**1)
                    # norm_low = torch.mean(F.relu(torch.log(torch.ones(z_selected.shape[0])/self.L)-torch.log(sigma_min))**1)
                    norm_up = torch.mean(F.relu(sigma_max - torch.ones(z_selected.shape[0]) * self.L) ** 2)
                    norm_low = torch.mean(F.relu(torch.ones(z_selected.shape[0]) / self.L - sigma_min) ** 2)
                    grad_loss = norm_up + norm_low
                    train_loss = recon_loss + self.lam_grad * grad_loss + self.lam_geo * geo_loss

                elif self.reg_grad == 'bi-lipschitz_enc':
                    jac_enc = func.vmap(func.jacrev(model.encoder))(x[indices])
                    jac_enc = jac_enc.view(jac_enc.size(0), -1, jac_enc.size(-1))
                    # Jacobian multiply by tangent space
                    jac_enc = torch.bmm(jac_enc, self.tangent_spaces[ids])
                    
                    norm_up = torch.mean(F.relu(torch.linalg.matrix_norm(jac_enc, ord=2) - torch.ones(z_selected.shape[0]) * self.L) ** 2)
                    norm_low = torch.mean(F.relu(torch.ones(z_selected.shape[0]) / self.L - torch.linalg.matrix_norm(jac_enc, ord=-2)) ** 2)
                    grad_loss = norm_up + norm_low
                    train_loss = recon_loss + self.lam_grad * grad_loss + self.lam_geo * geo_loss

                elif self.reg_grad == 'geometric':
                    jac_dec = func.vmap(func.jacfwd(model.decoder))(z_selected)
                    jac_dec = jac_dec.view(jac_dec.size(0), -1, jac_dec.size(-1))
                    riem_metric_dec = torch.bmm(jac_dec.transpose(1, 2), jac_dec)
                    # Calculate the logarithm of the generalized jacobian determinant
                    log_dets = torch.logdet(riem_metric_dec)
                    # Replace nan values with a small number
                    eps = 1e-9
                    torch.nan_to_num(log_dets, nan=eps, posinf=eps, neginf=eps)
                    # Calculate the variance of the logarithm of the generalized jacobian determinant
                    # grad_loss = torch.var(log_dets)
                    grad_loss = torch.mean((log_dets - 1) ** 2)
                    train_loss = recon_loss + self.lam_grad * grad_loss + self.lam_geo * geo_loss

                else:
                    raise ValueError("Invalid regularization type")

                train_loss.backward()
                self.optim.step()
            self.scheduler.step()

            # Evaluation phase
            model.eval()
            for x, _, ids in self.test_loader:
                z_hat = model.encoder(x)
                x_hat = model.decoder(z_hat)
                test_recon_loss = criterion(x_hat, x)

            if verbose:
                print("Epoch: [{}/{}], \tTrain Loss: {:.6f}\t geo loss: {:.6f}\t grad loss: {:.6f}\n"
                    "\t\t\tTest Reconstruction Loss: {:.6f}\n".format(
                    epoch + 1,
                    epochs,
                    train_loss.data.item(),
                    geo_loss.data.item(),
                    grad_loss.data.item(),
                    test_recon_loss.data.item()
                ))
                current_lr = self.optim.param_groups[0]['lr']
                print(f'Epoch {epoch}, Current LR: {current_lr}')

            if self.return_loss:
                mse_loss_list.append(recon_loss.data.item())
                geo_loss_list.append(geo_loss.data.item())
                grad_loss_list.append(grad_loss.data.item())

        if self.return_loss:
            return mse_loss_list, geo_loss_list, grad_loss_list
       
      


class GRAETrainer(BasicTrainer):

    def __init__(self, target_embedding, lam, optimizer, scheduler=None, device='cpu'):
        super(GRAETrainer, self).__init__(optimizer=optimizer, scheduler=scheduler, device=device)
        self.lam = torch.as_tensor(lam, device=self.device, dtype=torch.float)  # regularization coefficient
        self.target_embedding = torch.as_tensor(target_embedding, device=self.device, dtype=torch.float)  # target embedding matrix

    def train(self, model, epochs, verbose=True):
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            # Training phase
            for x, _, idx in self.train_loader:
                model.train()
                model.zero_grad()

                # Forward pass
                z = model.encoder(x)
                x_hat = model(x)

                # Calculate losses
                recon_loss = criterion(x_hat, x)
                reg_loss = criterion(z, self.target_embedding[idx])
                train_loss = recon_loss + self.lam * reg_loss

                # Backward pass and optimization
                train_loss.backward()
                self.optim.step()
            self.scheduler.step()

            # Evaluation phase
            model.eval()
            for x, _, _ in self.test_loader:
                x_hat = model(x)
                test_recon_loss = criterion(x_hat, x)

            if verbose:
                print("Epoch: [{}/{}], \tTrain Loss: {:.6f}\t Regularization Loss: {:.6f}\n"
                      "\t\t\tTest Reconstruction Loss: {:.6f}\n".format(
                    epoch + 1,
                    epochs,
                    train_loss.data.item(),
                    reg_loss.data.item(),
                    test_recon_loss.data.item()
                ))
                current_lr = self.optim.param_groups[0]['lr']
                print(f'Epoch {epoch + 1}, Current LR: {current_lr}')




class IRAETrainer(BasicTrainer):
    """
    Isometric regularized autoencoder trainer.
    """

    def __init__(self, lambda_iso, eta, optimizer, metric='identity', scheduler=None, device='cpu'):
        super(IRAETrainer, self).__init__(optimizer=optimizer, scheduler=scheduler, device=device)
        self.eta = torch.tensor(eta, device=device, dtype=torch.float)
        self.lam_iso = torch.tensor(lambda_iso, device=device, dtype=torch.float)
        self.reg = utils.regularizations.IsometricRegularization()
        self.device = device

    def train(self, model, epochs, verbose=True):
        criterion = nn.MSELoss()
        criterion.to(self.device)

        for epoch in range(epochs):
            for x, _, ids in self.train_loader:
                model.train()
                model.zero_grad()

                z_hat = model.encoder(x)
                x_hat = model.decoder(z_hat)

                reg_loss = self.reg(z_hat, model.decoder, self.eta)
                train_loss = criterion(x_hat, x) + self.lam_iso * reg_loss
                train_loss.backward()
                self.optim.step()
            self.scheduler.step()

            model.encoder.eval()
            model.decoder.eval()
            for x, _, ids in self.test_loader:
                z_hat = model.encoder(x)
                x_hat = model.decoder(z_hat)
                test_recon_loss = criterion(x_hat, x)

            if verbose:
                print("Epoch: [{}/{}], \tTrain Loss: {:.6f}, \t Regularization: {:.6f}, \tTest Reconstruction Loss: {:.6f}".format(
                    epoch + 1,
                    epochs,
                    train_loss.data.item(),
                    reg_loss.data.item(),
                    test_recon_loss.data.item()
                ))
                # current_lr = self.optim.param_groups[0]['lr']
                # print(f'Epoch {epoch + 1}, Current LR: {current_lr}')




# Patch function definition
def patch_diffusion_map_robust():
    """Patch DiffusionMap eigenvalue computation method"""
    
    def robust_make_diffusion_coords(self, L):
        """Replace original _make_diffusion_coords method"""
        
        # Strategy 1: Try ARPACK with increased tolerance and iterations
        try:
            # Increase maxiter and reduce tolerance for better convergence
            evals, evecs = spsl.eigs(L, k=(self.n_evecs+1), which='LR', 
                                    maxiter=10000, tol=1e-6)
            ix = evals.argsort()[::-1][1:]
            evals = np.real(evals[ix])
            evecs = np.real(evecs[:, ix])
            dmap = np.dot(evecs, np.diag(np.sqrt(1. / evals)))
            return dmap, evecs, evals
        except spsl.ArpackNoConvergence as e:
            warnings.warn("ARPACK did not converge, trying alternative methods...")
            pass
        
        # Strategy 2: Try eigsh (for symmetric matrices) if L is symmetric
        # if np.allclose(L.toarray(), L.toarray().T, rtol=1e-10):
        try:
            print("Using eigsh for symmetric matrix...")
            # Use eigsh for symmetric matrices (more stable)
            evals, evecs = spsl.eigsh(L, k=(self.n_evecs+1), which='LA',
                                    maxiter=10000, tol=1e-6)
            ix = evals.argsort()[::-1][1:]
            evals = np.real(evals[ix])
            evecs = np.real(evecs[:, ix])
            dmap = np.dot(evecs, np.diag(np.sqrt(1. / evals)))
            return dmap, evecs, evals
        except spsl.ArpackNoConvergence:
            pass
        
        # Strategy 3: Convert to dense and use standard eigenvalue decomposition
        try:
            if L.shape[0] < 2000:  # Only for reasonably sized matrices
                L_dense = L.toarray()
                evals, evecs = eigh(L_dense)
                # Sort eigenvalues in descending order and skip the first one
                ix = evals.argsort()[::-1][1:self.n_evecs+1]
                evals = evals[ix]
                evecs = evecs[:, ix]
                dmap = np.dot(evecs, np.diag(np.sqrt(1. / evals)))
                return dmap, evecs, evals
        except Exception as e:
            warnings.warn(f"Dense eigenvalue decomposition failed: {e}")
        
        # Strategy 4: Use shift-invert mode with sigma close to 0
        try:
            # Use shift-invert mode to find eigenvalues near 0
            evals, evecs = spsl.eigs(L, k=(self.n_evecs+1), sigma=1e-6, which='LM',
                                    maxiter=10000, tol=1e-6)
            ix = evals.argsort()[::-1][1:]
            evals = np.real(evals[ix])
            evecs = np.real(evecs[:, ix])
            dmap = np.dot(evecs, np.diag(np.sqrt(1. / evals)))
            return dmap, evecs, evals
        except spsl.ArpackNoConvergence:
            pass
        
        # Strategy 5: Reduce the number of requested eigenvalues
        if self.n_evecs > 2:
            try:
                # Try with fewer eigenvalues
                n_reduced = min(self.n_evecs // 2, 2)
                evals, evecs = spsl.eigs(L, k=(n_reduced+1), which='LR',
                                        maxiter=10000, tol=1e-6)
                ix = evals.argsort()[::-1][1:]
                evals = np.real(evals[ix])
                evecs = np.real(evecs[:, ix])
                dmap = np.dot(evecs, np.diag(np.sqrt(1. / evals)))
                warnings.warn(f"Reduced number of eigenvalues to {n_reduced}")
                return dmap, evecs, evals
            except spsl.ArpackNoConvergence:
                pass
        
        # If all strategies fail, raise an informative error
        raise RuntimeError(
            "All eigenvalue computation strategies failed. "
            "This might indicate issues with the Laplacian matrix construction. "
            "Try reducing n_components, adjusting epsilon/alpha parameters, "
            "or preprocessing your data (e.g., removing outliers, normalizing)."
        )

    
    # Apply patch
    dm.DiffusionMap._make_diffusion_coords = robust_make_diffusion_coords
    print("✓ DiffusionMap patch applied")

# Apply patch immediately
# patch_diffusion_map_robust()

class DiffusionNetTrainer(BasicTrainer):

    def __init__(self, lam=1, eta=1, n_components=2, n_neighbors=5, alpha=0.1, epsilon='bgh_generous', random_state=42,
                 optimizer=None, scheduler=None, device='cpu'):
        """
        Init.

        Args:
            lam(float): Regularization factor for the coordinate constraint.
            eta(float): Regularization factor for the EV constraint.
            n_neighbors(int): The size of local neighborhood used to build the neighborhood graph.
            alpha(float): Exponent to be used for the left normalization in constructing the diffusion map.
            epsilon(Any):  Method for choosing the epsilon. See scikit-learn NearestNeighbors class for details.
            **kwargs: All other keyword arguments are passed to the AE parent class.
        """
        super(DiffusionNetTrainer, self).__init__(optimizer=optimizer, scheduler=scheduler, device=device)
        self.lam = torch.tensor(lam, device=device, dtype=torch.float)
        self.eta = torch.tensor(eta, device=device, dtype=torch.float)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.epsilon = epsilon
        self.random_state = random_state

    def fit(self, x):
        """
        Fit model to data.
        Args: x(np.array): Dataset to fit.
        """
        self.batch_size = x.shape[0]
        x_np = x.cpu().numpy().reshape(self.batch_size, -1)

        # Reduce dimensionality for faster kernel computations. We do the same with PHATE and UMAP.
        if x_np.shape[1] > 100 and x_np.shape[0] > 1000:
            print('Computing PCA before running DM...')
            x_np = PCA(n_components=100, random_state=self.random_state).fit_transform(x_np)

        neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=self.n_components, alpha=self.alpha, epsilon=self.epsilon,
                                              k=self.n_neighbors, neighbor_params=neighbor_params)
        dmap = mydmap.fit_transform(x_np)

        self.z = torch.tensor(dmap).float().to(self.device)

        self.Evectors = torch.from_numpy(mydmap.evecs).float().to(self.device)
        self.Evalues = torch.from_numpy(mydmap.evals).float().to(self.device)

        # Potential matrix sparse form
        P = scipy.sparse.coo_matrix(mydmap.L.todense())
        values = P.data
        indices = np.vstack((P.row, P.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        # self.P = torch.sparse.FloatTensor(i, v).float()
        self.P = torch.sparse_coo_tensor(i, v, dtype=torch.float32)

        # Identity matrix sparse
        I_n = scipy.sparse.coo_matrix(np.eye(self.batch_size))
        values = I_n.data
        indices = np.vstack((I_n.row, I_n.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        # self.I_t = torch.sparse.FloatTensor(i, v).float()
        self.I_t = torch.sparse_coo_tensor(i, v, dtype=torch.float32)

    def compute_loss(self, x, x_hat, z, idx):
        """
        Compute diffusion-based loss.
        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.
        """
        criterion = nn.MSELoss()
        rec_loss = criterion(x, x_hat)
        coord_loss = criterion(z, self.z[idx])
        Ev_loss = (
            torch.mean(torch.pow(torch.mm((self.P.to_dense().to(z) - self.Evalues[0].to(z) * self.I_t.to_dense().to(z)), z[:, 0].view(self.batch_size, 1)), 2)) +
            torch.mean(torch.pow(torch.mm((self.P.to_dense().to(z) - self.Evalues[1].to(z) * self.I_t.to_dense().to(z)), z[:, 1].view(self.batch_size, 1)), 2))
        )

        loss = rec_loss + self.lam * coord_loss + self.eta * Ev_loss
        return loss, rec_loss, coord_loss, Ev_loss

    def train(self, model, epochs, verbose=True):
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            for x, _, ids in self.train_loader:
                # self.fit(x)
                model.train()
                model.zero_grad()

                z_hat = model.encoder(x)
                x_hat = model.decoder(z_hat)

                train_total_loss, _, coord_loss, Ev_loss = self.compute_loss(x, x_hat, z_hat, ids)
                train_total_loss.backward()
                self.optim.step()
            self.scheduler.step()

            model.eval()
            for x, _, ids in self.test_loader:
                z_hat = model.encoder(x)
                x_hat = model.decoder(z_hat)
                test_recon_loss = criterion(x_hat, x)

            if verbose:
                print("Epoch: [{}/{}], \tTrain Loss: {:.6f}, \tCoord Loss: {:.6f}, \tEV Loss: {:.6f}, \tTest Reconstruction Loss: {:.6f}".format(
                    epoch + 1,
                    epochs,
                    train_total_loss.data.item(),
                    coord_loss.data.item(),
                    Ev_loss.data.item(),
                    test_recon_loss.data.item()
                ))




class PCAAETrainer(BasicTrainer):
    def __init__(self, lambda_rec=1.0, lambda_cov=1.0, optimizer=None, scheduler=None, device='cpu'):
        super(PCAAETrainer, self).__init__(optimizer=optimizer, scheduler=scheduler, device=device)
        self.lambda_rec = torch.tensor(lambda_rec, device=device, dtype=torch.float)
        self.lambda_cov = torch.tensor(lambda_cov, device=device, dtype=torch.float)
        self.latent_spaces = []  # Store outputs of previous encoders

    def cov_loss(self, z):
        if z.shape[1] < 2:
            return torch.tensor(0.0, device=z.device)
        last_col = z[:, -1]
        loss = 0.0
        for i in range(z.shape[1] - 1):
            loss += (torch.mean(z[:, i] * last_col)) ** 2
        return loss / (z.shape[1] - 1)

    def train(self, encoder, decoder, optim, scheduler, step, epochs, verbose=True):
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            for x, _, _ in self.train_loader:
                encoder.train()
                decoder.train()
                encoder.zero_grad()
                decoder.zero_grad()

                # Compute latent representations
                with torch.no_grad():
                    prev_z_list = [prev_enc(x) for prev_enc in self.latent_spaces]
                z = encoder(x)
                if prev_z_list:
                    z_all = torch.cat(prev_z_list + [z], dim=1)
                else:
                    z_all = z

                x_hat = decoder(z_all)
                recon_loss = criterion(x_hat, x)
                cov_reg = self.cov_loss(z_all)
                # cov_reg = self.cov_loss(z) if self.latent_spaces else 0.0
                total_loss = self.lambda_rec * recon_loss + self.lambda_cov * cov_reg

                total_loss.backward()
                optim.step()

            scheduler.step()

            # Evaluation
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                for x, _, _ in self.test_loader:
                    prev_z_list = [prev_enc(x) for prev_enc in self.latent_spaces]
                    z = encoder(x)
                    if prev_z_list:
                        z_all = torch.cat(prev_z_list + [z], dim=1)
                    else:
                        z_all = z
                    x_hat = decoder(z_all)
                    test_loss = criterion(x_hat, x)

            if verbose:
                print(f"[PCA-AE Step {step}] Epoch {epoch+1}/{epochs}, \tTrain Loss: {total_loss.item():.6f}, Cov Loss: {cov_reg:.6f}, Test Loss: {test_loss.item():.6f}")

        # After training this encoder, store it
        self.latent_spaces.append(copy.deepcopy(encoder))