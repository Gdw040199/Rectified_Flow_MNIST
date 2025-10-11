import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# MiNi U-Net model for image-to-image tasks
class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=16, downsample=False):
        super(DownLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.fc = nn.Linear(time_emb_dim, in_channels) # [B, dim] -> [B, in_channels]
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None
        # downsample at the end of the block
        self.downsample = downsample
        if downsample:
            self.downsample_layer = nn.MaxPool2d(2)
        self.in_channels = in_channels
        
    def forward(self, x, t):
        res = x
        x += self.fc(t)[:, :, None, None] # [B, in_channels, H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        if self.shortcut is not None:
            res = self.shortcut(res)
        x = x + res
        if self.downsample:
            x = self.downsample_layer(x)
        return x

class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=16, upsample=False):
        super(UpLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.fc = nn.Linear(time_emb_dim, in_channels) # [B, dim] -> [B, in_channels]
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, x, t_emb):
        if self.upsample:
            x = self.upsample_layer(x)
        res = x
        x += self.fc(t_emb)[:, :, None, None] # [B, in_channels, H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        if self.shortcut is not None:
            res = self.shortcut(res)
        x = x + res
        return x

class MiddleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=16):
        super(MiddleLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.fc = nn.Linear(time_emb_dim, in_channels) # [B, dim] -> [B, in_channels]
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x, t):
        res = x
        x += self.fc(t)[:, :, None, None] # [B, in_channels, H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        if self.shortcut is not None:
            res = self.shortcut(res)
        x = x + res
        return x
    
class MiniUNet(nn.Module):

    def __init__(self, base_channels=16, time_emb_dim=None):
        super(MiniUNet, self).__init__()
        if time_emb_dim is None:
            self.time_emb_dim = base_channels
        self.base_channels = base_channels
        self.conv_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)
        self.down1 = nn.ModuleList([
            DownLayer(base_channels, base_channels * 2, time_emb_dim=self.time_emb_dim, downsample=False),
            DownLayer(base_channels * 2, base_channels * 2, time_emb_dim=self.time_emb_dim)
        ])
        self.maxpool1 = nn.MaxPool2d(2)
        self.down2 = nn.ModuleList([
            DownLayer(base_channels * 2, base_channels * 4, time_emb_dim=self.time_emb_dim, downsample=False),
            DownLayer(base_channels * 4, base_channels * 4, time_emb_dim=self.time_emb_dim)
        ])
        self.maxpool2 = nn.MaxPool2d(2)
        self.middle = MiddleLayer(base_channels * 4, base_channels * 4, time_emb_dim=self.time_emb_dim)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.up1 = nn.ModuleList([
            UpLayer(base_channels * 8, base_channels * 2, time_emb_dim=self.time_emb_dim, upsample=False),
            UpLayer(base_channels * 2, base_channels * 2, time_emb_dim=self.time_emb_dim)
        ])
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.up2 = nn.ModuleList([
            UpLayer(base_channels * 4, base_channels, time_emb_dim=self.time_emb_dim, upsample=False),
            UpLayer(base_channels, base_channels, time_emb_dim=self.time_emb_dim)
        ])
        self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=1, padding=0)


    # May cause instability and not clear improvement
    # Time and label embedding functions
    """
    def time_emb(self, t, dim):
        t = t * 1000 # scale to [0, 1000]
        freqs= torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)
    
    def label_emb(self, y, dim):
        y = y * 1000 # scale to [0, 1000]
        freqs= torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(y.device)
        sin_emb = torch.sin(y[:, None] / freqs)
        cos_emb = torch.cos(y[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)
    """

    def time_emb(self, t, dim): 
        """
        Args:
            t (torch.Tensor): [B] time in [0, 1]
            dim (int): dimension of time embedding
        Returns:
            time_emb (torch.Tensor): [B, dim] time embedding
        Explanation:
            This function generates a sinusoidal time embedding similar to the one used in the Transformer model.
            The time t is first clamped to [0, 1] to ensure valid input. Then, a set of frequencies is computed
            using an exponential scale. The time embedding is formed by concatenating sine and cosine functions
            of the scaled time values. If the dimension is odd, an extra zero padding is added to maintain the dimension.
        """
        t = torch.clamp(t, 0.0, 1.0)
        half_dim = dim // 2  
        emb_coeff = -math.log(1000) / (half_dim - 1 + 1e-5)
        freqs = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * emb_coeff)  # [half_dim]
        time_emb = t.unsqueeze(1) * freqs.unsqueeze(0)  # [B, half_dim]
        time_emb = torch.cat([torch.sin(time_emb), torch.cos(time_emb)], dim=1)  # [B, dim]
        if dim % 2 == 1:
            time_emb = F.pad(time_emb, (0, 1), mode='constant', value=0)  # [B, dim]
        
        return time_emb

    def label_emb(self, y, dim):
        """
        Args:
            y (torch.Tensor): [B] label in {0, 1, ..., 9}, -1 means unconditioned
            dim (int): dimension of label embedding
        Returns:
            label_emb (torch.Tensor): [B, dim] label embedding
        Explanation:    
            This function generates a sinusoidal label embedding. The label y is first masked to handle
            unconditioned cases (where y = -1). The label is then scaled to the range [0, 2π/9] to map the
            discrete labels to a continuous space. A set of frequencies is computed using a logarithmic scale.
            The label embedding is formed by concatenating sine and cosine functions of the scaled label values.
            The embedding is masked to ensure that unconditioned labels result in a zero embedding.
        """
        mask = (y >= 0).float().unsqueeze(1)  # [B, 1]
        y_cont = y.float() * (2 * math.pi / 9)  # [B]
        y_cont = y_cont * mask.squeeze(1) 
        min_freq = 0.1
        max_freq = 10.0
        freqs = torch.logspace(math.log10(min_freq), math.log10(max_freq), steps=dim // 2, device=y.device, dtype=torch.float32)  # [dim//2]
        sin_emb = torch.sin(y_cont.unsqueeze(1) * freqs.unsqueeze(0))  # [B, dim//2]
        cos_emb = torch.cos(sin_emb)  # [B, dim//2]
        label_emb = torch.cat([sin_emb, cos_emb], dim=1)  # [B, dim]
        label_emb = label_emb * mask
        
        return label_emb

    def forward(self, x, t, y=None):
        """
        Args:
            x (torch.Tensor): input image tensor [B, 1, 28, 28]
            t (torch.Tensor): time tensor [B]
            y (torch.Tensor, optional): label tensor [B]
        """
        x = self.conv_in(x)
        t = self.time_emb(t, self.base_channels)
        
        # add the label embedding and time embedding
        if y is not None:
            if len(y.shape) == 1:
                # -1 means unconditioned, then set embedding to zero
                y_emb = self.label_emb(y, self.base_channels)
                y_emb[y == -1] = 0.0
                t = t + y_emb
            else:
                pass

        # downsample
        for layer in self.down1:
            x = layer(x, t)
        x1 = x
        x = self.maxpool1(x)
        for layer in self.down2:
            x = layer(x, t)
        x2 = x
        x = self.maxpool2(x)

        # middle
        x = self.middle(x, t)

        # upsample
        x = torch.cat([self.upsample1(x), x2], dim=1)
        for layer in self.up1:
            x = layer(x, t)
        x = torch.cat([self.upsample2(x), x1], dim=1)
        for layer in self.up2:
            x = layer(x, t)

        x = self.conv_out(x)
        return x
