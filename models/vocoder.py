"""
HiFi-GAN Vocoder

v2 Updates:
- Domyślny sample_rate: 32000 (zamiast 22050)
- Dostosowane upsample_rates dla 32kHz
- Predefiniowane konfiguracje dla różnych sample rates

Vocoder do konwersji mel-spektrogramów na audio
Bazowany na HiFi-GAN v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from typing import List


LRELU_SLOPE = 0.1

# v2: Predefined configurations for different sample rates
VOCODER_CONFIGS = {
    # 22050 Hz: hop_length=256 → upsample ratio = 256
    22050: {
        'hop_length': 256,
        'upsample_rates': [8, 8, 2, 2],      # 8*8*2*2 = 256
        'upsample_kernel_sizes': [16, 16, 4, 4],
    },
    # 32000 Hz: hop_length=320 → upsample ratio = 320
    32000: {
        'hop_length': 320,
        'upsample_rates': [8, 8, 4, 2, 2],   # 8*8*4*2*2 = 256... ale potrzebujemy 320
        # Alternatywnie: 10, 8, 4, czyli 10*8*4 = 320
        'upsample_rates': [10, 8, 4],         # 10*8*4 = 320 ✓
        'upsample_kernel_sizes': [20, 16, 8],
    },
    # 44100 Hz: hop_length=512 → upsample ratio = 512
    44100: {
        'hop_length': 512,
        'upsample_rates': [8, 8, 4, 2],       # 8*8*4*2 = 512
        'upsample_kernel_sizes': [16, 16, 8, 4],
    },
    # 48000 Hz: hop_length=480 → upsample ratio = 480
    48000: {
        'hop_length': 480,
        'upsample_rates': [8, 6, 5, 2],       # 8*6*5*2 = 480
        'upsample_kernel_sizes': [16, 12, 10, 4],
    },
}


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):
    """Residual Block Type 1"""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=d, padding=get_padding(kernel_size, d)
            ))
            for d in dilation
        ])
        self.convs1.apply(init_weights)
        
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=1, padding=get_padding(kernel_size, 1)
            ))
            for _ in dilation
        ])
        self.convs2.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x
    
    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    """Residual Block Type 2 (simpler)"""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=d, padding=get_padding(kernel_size, d)
            ))
            for d in dilation
        ])
        self.convs.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x
    
    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN Generator (Vocoder)
    
    v2 Updates:
    - Domyślne parametry dla 32kHz
    - Metoda from_sample_rate() do automatycznej konfiguracji
    
    Konwertuje mel-spektrogramy na waveformy audio
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        upsample_rates: List[int] = [10, 8, 4],       # v2: dla 32kHz (10*8*4=320)
        upsample_kernel_sizes: List[int] = [20, 16, 8],  # v2: dla 32kHz
        upsample_initial_channel: int = 512,
        resblock_type: str = "1",
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        sample_rate: int = 32000,  # v2: Informacyjne, dla metadanych
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        # Calculate total upsample ratio
        self.upsample_ratio = 1
        for r in upsample_rates:
            self.upsample_ratio *= r
        
        # Initial convolution
        self.conv_pre = weight_norm(nn.Conv1d(
            n_mels, upsample_initial_channel, 7, 1, padding=3
        ))
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                upsample_initial_channel // (2 ** i),
                ch,
                k, u,
                padding=(k - u) // 2
            )))
        
        # Residual blocks
        ResBlock = ResBlock1 if resblock_type == "1" else ResBlock2
        self.resblocks = nn.ModuleList()
        
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))
        
        # Final convolution
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
    
    @classmethod
    def from_sample_rate(cls, sample_rate: int = 32000, n_mels: int = 128, **kwargs):
        """
        Tworzy vocoder z automatyczną konfiguracją dla danego sample rate.
        
        Args:
            sample_rate: Docelowy sample rate (22050, 32000, 44100, 48000)
            n_mels: Liczba mel bins
            **kwargs: Dodatkowe argumenty
            
        Returns:
            HiFiGANGenerator skonfigurowany dla danego sample rate
        """
        if sample_rate not in VOCODER_CONFIGS:
            print(f"⚠️ Sample rate {sample_rate} not in presets, using closest")
            # Find closest
            closest = min(VOCODER_CONFIGS.keys(), key=lambda x: abs(x - sample_rate))
            sample_rate = closest
        
        config = VOCODER_CONFIGS[sample_rate]
        
        return cls(
            n_mels=n_mels,
            upsample_rates=config['upsample_rates'],
            upsample_kernel_sizes=config['upsample_kernel_sizes'],
            sample_rate=sample_rate,
            **kwargs,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: mel-spectrogram [B, n_mels, T]
            
        Returns:
            audio: waveform [B, 1, T * prod(upsample_rates)]
        """
        x = self.conv_pre(x)
        
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class PeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (MPD) - single period"""
    
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
    
    def forward(self, x: torch.Tensor):
        fmap = []
        
        # Reshape to 2D
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator"""
    
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])
    
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator (MSD) - single scale"""
    
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else nn.utils.spectral_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))
    
    def forward(self, x: torch.Tensor):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator"""
    
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])
    
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    """Feature matching loss"""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Discriminator loss (LS-GAN)"""
    loss = 0
    r_losses = []
    g_losses = []
    
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """Generator adversarial loss"""
    loss = 0
    gen_losses = []
    
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    
    return loss, gen_losses


class HiFiGAN(nn.Module):
    """
    Kompletny HiFi-GAN z generatorem i dyskryminatorami
    
    v2 Updates:
    - from_sample_rate() dla automatycznej konfiguracji
    - Domyślnie 32kHz
    
    Do treningu vocodera lub używany jako pretrained
    """
    
    def __init__(self, n_mels: int = 128, sample_rate: int = 32000):
        super().__init__()
        self.sample_rate = sample_rate
        self.generator = HiFiGANGenerator.from_sample_rate(sample_rate, n_mels)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate audio from mel-spectrogram"""
        return self.generator(mel)
    
    @torch.no_grad()
    def inference(self, mel: torch.Tensor) -> torch.Tensor:
        """Inference mode - no gradients"""
        self.generator.eval()
        return self.generator(mel)
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, n_mels: int = 128, sample_rate: int = 32000):
        """Load pretrained vocoder"""
        model = cls(n_mels=n_mels, sample_rate=sample_rate)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.generator.load_state_dict(checkpoint['generator'])
        return model
    
    @classmethod
    def from_sample_rate(cls, sample_rate: int = 32000, n_mels: int = 128):
        """Tworzy HiFi-GAN skonfigurowany dla danego sample rate"""
        return cls(n_mels=n_mels, sample_rate=sample_rate)


# Can also use pretrained vocoder from huggingface/speechbrain
class PretrainedVocoder:
    """
    Wrapper dla pretrenowanych vocoderów
    
    Możliwe opcje:
    - HiFi-GAN z LJ Speech
    - Vocos
    - BigVGAN
    """
    
    def __init__(self, model_name: str = "nvidia/hifigan-lj-speech"):
        try:
            from transformers import AutoModel
            self.vocoder = AutoModel.from_pretrained(model_name)
        except:
            print(f"Could not load {model_name}, using custom HiFi-GAN")
            self.vocoder = HiFiGANGenerator()
    
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        return self.vocoder(mel)


if __name__ == "__main__":
    print("="*60)
    print("🎵 Testing HiFi-GAN Vocoder v2")
    print("="*60)
    
    # v2: Test for 32kHz (default)
    print("\n📊 Testing 32kHz vocoder (default)...")
    vocoder_32k = HiFiGANGenerator.from_sample_rate(32000, n_mels=128)
    mel = torch.randn(2, 128, 100)  # [B, n_mels, time]
    audio = vocoder_32k(mel)
    print(f"  Input mel: {mel.shape}")
    print(f"  Output audio: {audio.shape}")
    print(f"  Upsample ratio: {vocoder_32k.upsample_ratio} (hop_length)")
    print(f"  Expected audio length: {mel.shape[-1] * vocoder_32k.upsample_ratio}")
    
    # v2: Test for different sample rates
    print("\n📊 Testing different sample rates...")
    for sr in [22050, 32000, 44100, 48000]:
        vocoder = HiFiGANGenerator.from_sample_rate(sr, n_mels=128)
        audio = vocoder(mel)
        params = sum(p.numel() for p in vocoder.parameters())
        print(f"  {sr}Hz: upsample={vocoder.upsample_ratio}, audio={audio.shape}, params={params:,}")
    
    # Test full HiFi-GAN
    print("\n📊 Testing full HiFi-GAN (with discriminators)...")
    hifigan = HiFiGAN(n_mels=128, sample_rate=32000)
    audio = hifigan(mel)
    print(f"  HiFi-GAN output: {audio.shape}")
    
    params = sum(p.numel() for p in hifigan.parameters())
    print(f"  Total parameters: {params:,} ({params/1e6:.1f}M)")
