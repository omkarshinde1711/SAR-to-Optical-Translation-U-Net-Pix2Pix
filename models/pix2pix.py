import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class UNetGenerator(nn.Module):
    """UNet generator for Pix2Pix (reusing UNet architecture)"""
    
    def __init__(self, in_channels=1, out_channels=3, base_features=64):
        super(UNetGenerator, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.enc4 = self._conv_block(base_features * 4, base_features * 8)
        self.enc5 = self._conv_block(base_features * 8, base_features * 8)
        self.enc6 = self._conv_block(base_features * 8, base_features * 8)
        self.enc7 = self._conv_block(base_features * 8, base_features * 8)
        self.enc8 = self._conv_block(base_features * 8, base_features * 8, use_dropout=False)
        
        # Decoder
        self.dec8 = self._upconv_block(base_features * 8, base_features * 8, use_dropout=True)
        self.dec7 = self._upconv_block(base_features * 16, base_features * 8, use_dropout=True)
        self.dec6 = self._upconv_block(base_features * 16, base_features * 8, use_dropout=True)
        self.dec5 = self._upconv_block(base_features * 16, base_features * 8)
        self.dec4 = self._upconv_block(base_features * 16, base_features * 4)
        self.dec3 = self._upconv_block(base_features * 8, base_features * 2)
        self.dec2 = self._upconv_block(base_features * 4, base_features)
        self.dec1 = self._upconv_block(base_features * 2, out_channels, use_tanh=True)
        
    def _conv_block(self, in_channels, out_channels, use_dropout=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def _upconv_block(self, in_channels, out_channels, use_dropout=False, use_tanh=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        ]
        if use_tanh:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.ReLU(inplace=True))
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        
        # Decoder with skip connections
        d8 = self.dec8(e8)
        d8 = torch.cat([d8, e7], dim=1)
        
        d7 = self.dec7(d8)
        d7 = torch.cat([d7, e6], dim=1)
        
        d6 = self.dec6(d7)
        d6 = torch.cat([d6, e5], dim=1)
        
        d5 = self.dec5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        
        d4 = self.dec4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        
        d1 = self.dec1(d2)
        
        return d1


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator (70x70 receptive field)"""
    
    def __init__(self, in_channels=4, base_features=64, use_spectral_norm=False):
        super(PatchGANDiscriminator, self).__init__()
        
        self.use_spectral_norm = use_spectral_norm
        
        # Build discriminator layers
        layers = []
        
        # First layer
        layers.append(self._conv_layer(in_channels, base_features, use_norm=False))
        
        # Middle layers
        layers.append(self._conv_layer(base_features, base_features * 2))
        layers.append(self._conv_layer(base_features * 2, base_features * 4))
        layers.append(self._conv_layer(base_features * 4, base_features * 8))
        
        # Final layer
        layers.append(nn.Conv2d(base_features * 8, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def _conv_layer(self, in_channels, out_channels, use_norm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        if self.use_spectral_norm and use_norm:
            conv = spectral_norm(conv)
        
        layers = [conv, nn.LeakyReLU(0.2, inplace=True)]
        
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class Pix2PixModel(nn.Module):
    """Complete Pix2Pix model with generator and discriminator"""
    
    def __init__(self, in_channels=1, out_channels=3, base_features=64, 
                 use_spectral_norm=False, warmstart_path=None):
        super(Pix2PixModel, self).__init__()
        
        self.generator = UNetGenerator(in_channels, out_channels, base_features)
        self.discriminator = PatchGANDiscriminator(
            in_channels + out_channels, base_features, use_spectral_norm
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Warm-start generator if path provided
        if warmstart_path:
            self._warmstart_generator(warmstart_path)
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _warmstart_generator(self, unet_checkpoint_path):
        """Initialize generator weights from UNet checkpoint"""
        try:
            checkpoint = torch.load(unet_checkpoint_path, map_location='cpu')
            if 'model' in checkpoint:
                unet_state = checkpoint['model']
            else:
                unet_state = checkpoint
            
            # Load compatible weights
            generator_state = self.generator.state_dict()
            loaded_weights = 0
            
            for name, param in unet_state.items():
                if name in generator_state and param.shape == generator_state[name].shape:
                    generator_state[name] = param
                    loaded_weights += 1
            
            self.generator.load_state_dict(generator_state)
            print(f"Warm-started generator with {loaded_weights} compatible weights from {unet_checkpoint_path}")
            
        except Exception as e:
            print(f"Warning: Could not warm-start generator from {unet_checkpoint_path}: {e}")
    
    def forward(self, x):
        """Forward pass through generator"""
        return self.generator(x)
    
    def discriminate(self, x, y):
        """Discriminate real/fake pairs"""
        return self.discriminator(torch.cat([x, y], dim=1))


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    
    def __init__(self, device='cuda'):
        super(VGGPerceptualLoss, self).__init__()
        
        # Load pre-trained VGG16
        import torchvision.models as models
        vgg = models.vgg16(pretrained=True).features[:16]  # Up to conv3_3
        vgg.eval()
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.vgg = vgg.to(device)
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        """Compute perceptual loss between prediction and target"""
        # Normalize to ImageNet stats for VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        
        return self.mse(pred_features, target_features)


def create_pix2pix_model(config):
    """Factory function to create Pix2Pix model from config"""
    model = Pix2PixModel(
        in_channels=config.get('in_channels', 1),
        out_channels=config.get('out_channels', 3),
        base_features=config.get('base_features', 64),
        use_spectral_norm=config.get('use_spectral_norm', False),
        warmstart_path=config.get('warmstart_path', None)
    )
    
    return model