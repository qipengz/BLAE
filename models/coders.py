import torch.nn as nn
import torch


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:
                hidden_layers.append(nn.Linear(input_dim, hidden_dim[0]))
                hidden_layers.append(nn.ELU())
            else:
                hidden_layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
                hidden_layers.append(nn.ELU())
        self.layer_out = nn.Linear(hidden_dim[-1], self.latent_dim)
        self.hidden = nn.Sequential(*hidden_layers)

    def forward(self, x):
        x = self.hidden(x)
        z = self.layer_out(x)
        return z


class Decoder(nn.Module):

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:
                hidden_layers.append(nn.Linear(self.latent_dim, hidden_dim[0]))
                hidden_layers.append(nn.ELU())
            else:
                hidden_layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
                hidden_layers.append(nn.ELU())
        self.layer_out = nn.Linear(hidden_dim[-1], output_dim)
        self.hidden = nn.Sequential(*hidden_layers)

    def forward(self, z):
        x = self.hidden(z)
        x = self.layer_out(x)
        return x


class ConvNet64(nn.Module):

    def __init__(
        self, in_chan=1, out_chan=64, nh=32, out_activation="linear", activation="relu"
    ):
        """nh: determines the numbers of conv filters"""
        super(ConvNet64, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh, kernel_size=4, bias=True, stride=2, padding=1)
        self.conv2 = nn.Conv2d(nh, nh, kernel_size=4, bias=True, stride=2, padding=1)
        self.conv3 = nn.Conv2d(nh, 2 * nh, kernel_size=4, bias=True, stride=2, padding=1)
        self.conv4 = nn.Conv2d(2 * nh, 2 * nh, kernel_size=4, bias=True, stride=2, padding=1)
        self.conv5 = nn.Conv2d(2 * nh, 4 * nh, kernel_size=4, bias=True, stride=1)
        self.fc1 = nn.Conv2d(4 * nh, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation
        self.activation = activation
        if activation == "relu":
            act = nn.ReLU
        elif activation == "leakyrelu":

            def act():
                return nn.LeakyReLU(negative_slope=0.2, inplace=True)

        else:
            raise ValueError

        layers = [
            self.conv1,
            act(),
            self.conv2,
            act(),
            self.conv3,
            act(),
            self.conv4,
            act(),
            self.conv5,
            act(),
            self.fc1,
        ]

        if self.out_activation == "tanh":
            layers.append(nn.Tanh())
        elif self.out_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        if len(x.size()) == 4:
            return x.squeeze(2).squeeze(2)
        elif len(x.size()) == 3:
            return x.squeeze(1).squeeze(1)


class DeConvNet64(nn.Module):

    def __init__(
        self,
        in_chan=1,
        out_chan=1,
        nh=32,
        out_activation="sigmoid",
        activation="relu",
    ):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet64, self).__init__()
        self.fc1 = nn.ConvTranspose2d(in_chan, 4 * nh, kernel_size=1, bias=True)
        self.conv1 = nn.ConvTranspose2d(4 * nh, 2 * nh, kernel_size=4, bias=True)
        self.conv2 = nn.ConvTranspose2d(2 * nh, 2 * nh, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv3 = nn.ConvTranspose2d(2 * nh, nh, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv4 = nn.ConvTranspose2d(nh, nh, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv5 = nn.ConvTranspose2d(nh, out_chan, kernel_size=4, stride=2, padding=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation
        self.activation = activation
        if activation == "relu":
            act = nn.ReLU
        elif activation == "leakyrelu":

            def act():
                return nn.LeakyReLU(negative_slope=0.2, inplace=True)

        else:
            raise ValueError

        layers = [
            self.fc1,
            act(),
            self.conv1,
            act(),
            self.conv2,
            act(),
            self.conv3,
            act(),
            self.conv4,
            act(),
            self.conv5,
        ]

        if self.out_activation == "tanh":
            layers.append(nn.Tanh())
        elif self.out_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.size()) == 1:
            x = x.unsqueeze(1).unsqueeze(1)
        elif len(x.size()) == 2:
            x = x.unsqueeze(2).unsqueeze(2)
        x = self.net(x)
        return x


class ConvEncoder_Tea(nn.Module):
    """
    Simple example: Convolution -> MaxPool -> Flatten -> MLP
    for an image input of shape [3, 76, 128].
    """

    def __init__(self, latent_size=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # => [32, 38, 64]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # => [64, 19, 32]
        )
        # flatten size = 64 * 19 * 32 = 38912
        self.fc = nn.Sequential(
            nn.Linear(64 * 19 * 32, 400),
            nn.ReLU(),
            nn.Linear(400, latent_size),
        )

    def forward(self, x):
        """
        x shape: (B,3,76,128)
        returns latent z: (B, latent_size)
        """
        z = self.conv(x)  # => (B,64,19,32)
        z = z.view(z.size(0), -1)  # flatten => (B, 38912)
        z = self.fc(z)  # => (B, latent_size)
        return z


class ConvDecoder_Tea(nn.Module):
    """
    Mirrors the ConvEncoder: MLP -> reshape -> conv-transpose 
    to reconstruct shape [3, 76, 128].
    """

    def __init__(self, latent_size=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, 400),
            nn.ReLU(),
            nn.Linear(400, 64 * 19 * 32),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # => [32, 38, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),  # => [3, 76, 128]
            nn.Sigmoid(),
        )

    def forward(self, z):
        """
        z shape: (B, latent_size)
        returns reconstructed x: (B,3,76,128)
        """
        x = self.fc(z)  # => (B, 64*19*32)
        x = x.view(-1, 64, 19, 32)  # => (B,64,19,32)
        x = self.deconv(x)  # => (B,3,76,128)
        return x


class ConvEncoder_object(nn.Module):

    def __init__(self, latent_size=2):
        super().__init__()
        # For 64x64: 2 conv blocks
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # => [32, 32, 32]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # => [64, 16, 16]
        )
        # Flatten => 64*16*16 = 16384
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, latent_size),
        )

    def forward(self, x):
        # x shape: (B,3,64,64)
        z = self.conv(x)
        z = z.view(z.size(0), -1)  # flatten
        z = self.fc(z)  # => (B, latent_size)
        return z


class ConvDecoder_object(nn.Module):

    def __init__(self, latent_size=2):
        super().__init__()
        # We'll reverse the flatten dimension 64*16*16 = 16384
        self.fc = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 16 * 16),
            nn.ReLU(),
        )
        # Deconvolution blocks
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # => [32, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),  # => [3, 64, 64]
            nn.Sigmoid(),
        )

    def forward(self, z):
        # z shape: (B, latent_size)
        x = self.fc(z)  # => (B, 64*16*16)
        x = x.view(-1, 64, 16, 16)  # => (B,64,16,16)
        x = self.deconv(x)  # => (B,3,64,64)
        return x


class ConvEncoder_MNIST(nn.Module):
    """
    Convolutional encoder for 28x28 MNIST handwritten digit images.
    """

    def __init__(self, latent_size=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input is 1 channel (grayscale)
            nn.ReLU(),
            nn.MaxPool2d(2),  # => [32, 14, 14]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # => [64, 7, 7]
        )
        # Flatten size = 64 * 7 * 7 = 3136
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  # Using smaller intermediate layer
            nn.ReLU(),
            nn.Linear(128, latent_size),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 1, 28, 28)
        Returns:
            z: Latent representation of shape (B, latent_size)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        z = self.conv(x)
        z = z.view(z.size(0), -1)  # Flatten
        z = self.fc(z)  # => (B, latent_size)
        return z


class ConvDecoder_MNIST(nn.Module):
    """
    Convolutional decoder for reconstructing 28x28 MNIST handwritten digit images.
    """

    def __init__(self, latent_size=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 7 * 7),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # => [32, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),  # => [1, 28, 28]
            nn.Sigmoid(),  # Pixel values between 0-1
        )

    def forward(self, z):
        """
        Args:
            z: Latent representation of shape (B, latent_size)
        Returns:
            x: Reconstructed image of shape (B, 1, 28, 28)
        """
        x = self.fc(z)  # => (B, 64*7*7)
        x = x.view(-1, 64, 7, 7)  # => (B, 64, 7, 7)
        x = self.deconv(x)  # => (B, 1, 28, 28)
        return x


class ConvEncoder_Bird(nn.Module):

    def __init__(self, latent_size=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # [3,384,384] → [16,384,384]
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4),  # → [16,96,96]
            nn.Conv2d(16, 64, kernel_size=3, padding=1),  # → [64,96,96]
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),  # → [64,48,48]
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 48 * 48, 200),
            nn.ReLU(),
            nn.Linear(200, latent_size),
        )

    def forward(self, x):
        z = self.conv(x)  # → (B, 64, 48, 48)
        z = z.view(z.size(0), -1)  # → (B, 49152)
        z = self.fc(z)  # → (B, latent_size)
        return z


class ConvDecoder_Bird(nn.Module):

    def __init__(self, latent_size=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, 200),
            nn.ReLU(),
            nn.Linear(200, 64 * 48 * 48),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),  # → [16,48,48]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=4),  # → [3,192,192]
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)  # → (B, 49152)
        x = x.view(-1, 64, 48, 48)  # → (B, 64, 48, 48)
        x = self.deconv(x)  # → (B, 3, 192, 192)
        return x


class ConvEncoder_CIFAR10(nn.Module):
    """
    Simplified convolutional encoder for 32x32 CIFAR-10 color images.
    """

    def __init__(self, latent_size=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input is 3 channels (RGB)
            nn.ReLU(),
            nn.MaxPool2d(2),  # => [32, 16, 16]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # => [64, 8, 8]
        )
        # Flatten size = 64 * 8 * 8 = 4096
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, latent_size),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, 32, 32)
        Returns:
            z: Latent representation of shape (B, latent_size)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        z = self.conv(x)
        z = z.view(z.size(0), -1)  # Flatten
        z = self.fc(z)  # => (B, latent_size)
        return z


class ConvDecoder_CIFAR10(nn.Module):
    """
    Simplified convolutional decoder for reconstructing 32x32 CIFAR-10 color images.
    """

    def __init__(self, latent_size=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 8 * 8),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # => [32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),  # => [3, 32, 32]
            nn.Sigmoid(),  # Pixel values between 0-1
        )

    def forward(self, z):
        """
        Args:
            z: Latent representation of shape (B, latent_size)
        Returns:
            x: Reconstructed image of shape (B, 3, 32, 32)
        """
        x = self.fc(z)  # => (B, 64*8*8)
        x = x.view(-1, 64, 8, 8)  # => (B, 64, 8, 8)
        x = self.deconv(x)  # => (B, 3, 32, 32)
        return x