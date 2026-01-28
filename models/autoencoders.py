from . import coders
import numpy as np
import torch.nn as nn
import torch.func as func
import torch
import utils.regularizations


class Autoencoder(nn.Module):
    
    def __init__(self, input_dim=3, hidden_dim=512, latent_dim=2):
        super(Autoencoder, self).__init__()
        self.encoder = coders.Encoder(input_dim, [hidden_dim] * 2, latent_dim)
        self.decoder = coders.Decoder(latent_dim, [hidden_dim] * 2, input_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def get_encoder_jacobian(self, x):
        jacobian = torch.autograd.functional.jacobian(self.encoder, x, create_graph=True)
        jacobian = jacobian.diagonal(dim1=0, dim2=2)
        return jacobian.permute(2, 0, 1)
    
    def get_decoder_jacobian(self, z):
        jacobian = torch.autograd.functional.jacobian(self.decoder, z, create_graph=True)
        jacobian = jacobian.diagonal(dim1=0, dim2=2)
        return jacobian.permute(2, 0, 1)
    
    def compute_reconstruction_loss(self, x):
        recon = self.forward(x)
        loss = nn.MSELoss()(recon, x)
        return loss
    
    def compute_regularized_loss(self, GeoD, lam, x, thresh=0.3):
        reg = utils.regularizations.InjectiveLoss(thresh=thresh)
        z = self.encode(x)
        loss = self.compute_reconstruction_loss(x) + lam * reg(GeoD, z)
        return loss
    
    def compute_isometric_loss(self, GeoD, lam_geo, lam_grad, x):
        z_hat = self.encoder(x)
        batch_size = z_hat.size(0)
        indices = torch.randperm(batch_size)[:round(batch_size * 1)]
        z_selected = z_hat[indices]
        geo_loss = self.compute_regularized_loss(GeoD, lam_geo, x)
        jac_dec = func.vmap(func.jacfwd(self.decoder))(z_selected)
        jac_dec = jac_dec.view(jac_dec.size(0), -1, jac_dec.size(-1))
        riem_metric_dec = torch.bmm(jac_dec.transpose(1, 2), jac_dec)
        grad_loss = torch.mean(torch.norm(riem_metric_dec - torch.eye(z_hat.shape[1]).repeat(z_selected.size(0), 1, 1),
                                          p='fro', dim=(1, 2)) ** 2)
        
        return geo_loss + lam_grad * grad_loss


class DspritesAutoencoder(nn.Module):
    
    def __init__(self, in_chan=1, out_chan=3, nh=32):
        super(DspritesAutoencoder, self).__init__()
        self.encoder = coders.ConvNet64(in_chan=in_chan, out_chan=out_chan, nh=nh, out_activation="linear", activation="relu")
        self.decoder = coders.DeConvNet64(in_chan=out_chan, out_chan=in_chan, nh=nh, out_activation="linear", activation="relu")

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def get_encoder_jacobian(self, x):
        jacobian = torch.autograd.functional.jacobian(self.encoder, x, create_graph=True)
        jacobian = jacobian.diagonal(dim1=0, dim2=2)
        return jacobian.permute(2, 0, 1)
    
    def get_decoder_jacobian(self, z):
        jacobian = torch.autograd.functional.jacobian(self.decoder, z, create_graph=True)
        jacobian = jacobian.diagonal(dim1=0, dim2=2)
        return jacobian.permute(2, 0, 1)


class TeapotAutoencoder(nn.Module):
    """
    An autoencoder that uses ConvEncoder_Tea and ConvDecoder_Tea.
    """
    
    def __init__(self, latent_size=16):
        super(TeapotAutoencoder, self).__init__()
        self.encoder = coders.ConvEncoder_Tea(latent_size=latent_size)
        self.decoder = coders.ConvDecoder_Tea(latent_size=latent_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def get_encoder_jacobian(self, x):
        """
        Returns the diagonal of the Jacobian wrt each input dimension, as done in your other classes.
        WARNING: For large images, this can be expensive. 
        """
        jacobian = torch.autograd.functional.jacobian(self.encoder, x, create_graph=True)
        # jacobian shape => (output_shape..., input_shape...), very large for images
        # The code below assumes your existing approach of extracting the diagonal. 
        # For conv-based models, you might want a different approach.
        jacobian = jacobian.diagonal(dim1=0, dim2=2)
        return jacobian.permute(2, 0, 1)

    def get_decoder_jacobian(self, z):
        jacobian = torch.autograd.functional.jacobian(self.decoder, z, create_graph=True)
        jacobian = jacobian.diagonal(dim1=0, dim2=2)
        return jacobian.permute(2, 0, 1)


class ObjectAutoencoder(nn.Module):
    """
    An autoencoder that uses ConvEncoder_Tea and ConvDecoder_Tea.
    """
    
    def __init__(self, latent_size=16):
        super(ObjectAutoencoder, self).__init__()
        self.encoder = coders.ConvEncoder_object(latent_size=latent_size)
        self.decoder = coders.ConvDecoder_object(latent_size=latent_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def get_encoder_jacobian(self, x):
        """
        Returns the diagonal of the Jacobian wrt each input dimension, as done in your other classes.
        WARNING: For large images, this can be expensive. 
        """
        jacobian = torch.autograd.functional.jacobian(self.encoder, x, create_graph=True)
        # jacobian shape => (output_shape..., input_shape...), very large for images
        # The code below assumes your existing approach of extracting the diagonal. 
        # For conv-based models, you might want a different approach.
        jacobian = jacobian.diagonal(dim1=0, dim2=2)
        return jacobian.permute(2, 0, 1)

    def get_decoder_jacobian(self, z):
        jacobian = torch.autograd.functional.jacobian(self.decoder, z, create_graph=True)
        jacobian = jacobian.diagonal(dim1=0, dim2=2)
        return jacobian.permute(2, 0, 1)
    

class MNISTAutoencoder(nn.Module):
    """
    Convolutional autoencoder for 28x28 MNIST digit images.
    """
    
    def __init__(self, latent_size=16):
        super(MNISTAutoencoder, self).__init__()
        self.encoder = coders.ConvEncoder_MNIST(latent_size=latent_size)
        self.decoder = coders.ConvDecoder_MNIST(latent_size=latent_size)

    def encode(self, x):
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent representation to input space."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through entire autoencoder."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon
    
    def save_model(self, path):
        """Save model state dict to file."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load model state dict from file."""
        self.load_state_dict(torch.load(path))

    def get_encoder_jacobian(self, x):
        """
        Compute diagonal of encoder Jacobian matrix.
        Warning: Expensive for large images, but more manageable for MNIST's 28x28.
        """
        jacobian = torch.autograd.functional.jacobian(self.encoder, x, create_graph=True)
        jacobian = jacobian.diagonal(dim1=0, dim2=2)
        return jacobian.permute(2, 0, 1)

    def get_decoder_jacobian(self, z):
        """Compute diagonal of decoder Jacobian matrix."""
        jacobian = torch.autograd.functional.jacobian(self.decoder, z, create_graph=True)
        jacobian = jacobian.diagonal(dim1=0, dim2=2)
        return jacobian.permute(2, 0, 1)
    

class BirdAutoencoder(nn.Module):
    
    def __init__(self, latent_size=16):
        super(BirdAutoencoder, self).__init__()
        self.encoder = coders.ConvEncoder_Bird(latent_size=latent_size)
        self.decoder = coders.ConvDecoder_Bird(latent_size=latent_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def get_encoder_jacobian(self, x):
        jacobian = torch.autograd.functional.jacobian(self.encoder, x, create_graph=True)
        jacobian = jacobian.diagonal(dim1=0, dim2=2)
        return jacobian.permute(2, 0, 1)

    def get_decoder_jacobian(self, z):
        jacobian = torch.autograd.functional.jacobian(self.decoder, z, create_graph=True)
        jacobian = jacobian.diagonal(dim1=0, dim2=2)
        return jacobian.permute(2, 0, 1)


class CIFAR10Autoencoder(nn.Module):
    """
    Convolutional autoencoder for 32x32 CIFAR-10 color images.
    """
    
    def __init__(self, latent_size=16):
        super(CIFAR10Autoencoder, self).__init__()
        self.encoder = coders.ConvEncoder_CIFAR10(latent_size=latent_size)
        self.decoder = coders.ConvDecoder_CIFAR10(latent_size=latent_size)

    def encode(self, x):
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent representation to input space."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through entire autoencoder."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon
    
    def save_model(self, path):
        """Save model state dict to file."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load model state dict from file."""
        self.load_state_dict(torch.load(path))

    def get_encoder_jacobian(self, x):
        """
        Compute diagonal of encoder Jacobian matrix.
        Warning: Expensive for large images, but more manageable for CIFAR-10's 32x32.
        """
        jacobian = torch.autograd.functional.jacobian(self.encoder, x, create_graph=True)
        jacobian = jacobian.diagonal(dim1=0, dim2=2)
        return jacobian.permute(2, 0, 1)

    def get_decoder_jacobian(self, z):
        """Compute diagonal of decoder Jacobian matrix."""
        jacobian = torch.autograd.functional.jacobian(self.decoder, z, create_graph=True)
        jacobian = jacobian.diagonal(dim1=0, dim2=2)
        return jacobian.permute(2, 0, 1)