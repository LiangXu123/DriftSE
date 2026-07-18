import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Use relative import for registry (assuming file location)
try:
    from .shared import BackboneRegistry
except ImportError:
    # Fallback if running standalone
    class BackboneRegistry:
        @staticmethod
        def register(name):
            def decorator(cls):
                return cls
            return decorator

# ==========================================
# 1. HELPER LAYERS (1D Versions)
# ==========================================
def get_timestep_embedding(timesteps, embedding_dim):
    """Sinusoidal embeddings for time t"""
    # Verify shape
    if len(timesteps.shape) == 0:
        timesteps = timesteps.unsqueeze(0)
    
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    return emb

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        
        # Add time embedding (broadcast over time)
        h += self.time_proj(self.act(t_emb))[:, :, None]
        
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

# ==========================================
# 2. MAIN MODEL: 1D U-Net for Latents
# ==========================================
@BackboneRegistry.register("latent_unet")
class LatentDriftingUNet(nn.Module):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--input_dim", type=int, default=128)
        parser.add_argument("--model_dim", type=int, default=128)
        parser.add_argument("--dim_mults", type=int, nargs='+', default=[1, 2, 4, 8])
        return parser

    def __init__(self, 
                 input_dim=128,   # EnCodec latent size
                 model_dim=128,   # Base channel count
                 dim_mults=(1, 2, 4, 8),
                 **kwargs):       # Absorb extra args from config
        super().__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        
        # 1. Input Projection
        # We take [Noisy_Latent; Condition_Latent] -> 2 * input_dim channels
        self.input_proj = nn.Conv1d(input_dim * 2, model_dim, 3, padding=1)
        
        # 2. Time Embedding
        time_dim = model_dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # 3. Down Sampling
        self.downs = nn.ModuleList()
        curr_dim = model_dim
        for mult in dim_mults:
            out_dim = model_dim * mult
            # Each level has ResBlock -> ResBlock -> Downsample
            # We bundle them to match forward loop unpacking logic
            self.downs.append(nn.ModuleList([
                ResidualBlock1D(curr_dim, out_dim, time_dim),
                ResidualBlock1D(out_dim, out_dim, time_dim),
                Downsample1D(out_dim)
            ]))
            curr_dim = out_dim

        # 4. Middle
        self.mid_block1 = ResidualBlock1D(curr_dim, curr_dim, time_dim)
        self.mid_block2 = ResidualBlock1D(curr_dim, curr_dim, time_dim)

        # 5. Up Sampling
        self.ups = nn.ModuleList()
        for mult in reversed(dim_mults):
            out_dim = int(model_dim * mult) # Ensure integer
            # Skip connection adds 'out_dim' channels (from Down path)
            # Input to Up block is 'curr_dim' (from previous layer)
            # We concat [h, skip]. skip has 'out_dim'. h has 'curr_dim'.
            # Wait! In Down path, output of level L sends 'out_dim' to next level.
            # And 'out_dim' is stored in skips.
            # In Up path, we process level L in reverse.
            # We receive 'curr_dim' from deeper level.
            # We upscale 'curr_dim'.
            # Then concat with skip (from Down level L).
            # Down level L output was 'out_dim'.
            # So concat is [upscaled_h, skip] -> 'curr_dim' + 'out_dim' ?
            # Logic check:
            # Down: curr=128 -> out=128. Skip=128. Downsampled.
            # Up: curr=256 -> out=128. Upscale. Skip=128. Concat=256+128?
            # Recheck Loop structure.
            
            # Down Loop:
            # mult=1: curr=128, out=128. (128->128). Skip=128. curr becomes 128.
            # mult=2: curr=128, out=256. (128->256). Skip=256. curr becomes 256.
            # ...
            
            # Mid: 1024.
            
            # Up Loop (reversed):
            # mult=8 (1024). out_dim=1024.
            # We are coming from Mid (1024).
            # This loop logic seems slightly off in snippet or standard Unet.
            # Standard Unet usually reduces dimension in Up blocks.
            # User snippet:
            # for mult in reversed(dim_mults):
            #    out_dim = model_dim * mult
            
            # If mults=[1, 2, 4, 8]. Reversed=[8, 4, 2, 1].
            # Iter 1 (mult=8): out_dim=1024. curr_dim (from Mid) is 1024.
            # Upsample1D(1024) -> 1024.
            # Skip is from Down layer corresponding to mult=8?
            # Down layer mult=8 produced output 1024.
            # So Skip=1024.
            # Concat: 1024+1024 = 2048.
            # ResBlock1D(2048, 1024).
            # ResBlock1D(1024, 1024).
            # curr_dim becomes 1024.
            
            # Iter 2 (mult=4): out_dim=512.
            # Upsample1D(1024) -> 1024? No, Upsample usually reduces channels?
            # Or keeps same?
            # layers.Upsample1D in user snippet: conv(dim, dim). Channel count preserved!
            # So input to next stage is 1024.
            # BUT we want 512!
            # The User snippet Upsample1D does NOT change channels.
            # So `input_dim` to ResBlock will be `1024 (upsampled) + 512 (skip)`.
            # ResBlock(1536, 512).
            # This works.
            
            # Let's verify `curr_dim` vs `out_dim`.
            # Before Loop, `curr_dim` is output of Mid Block (e.g. 1024).
            # Loop mult=8: out_dim=1024.
            # Upsample(1024).
            # Skip=1024.
            # ResBlock(1024+1024, 1024).
            # curr_dim = 1024.
            
            # Loop mult=4: out_dim=512.
            # Upsample(1024).
            # Skip=512.
            # ResBlock(1024+512, 512).
            # curr_dim = 512.
            
            # Loops ...
            
            self.ups.append(nn.ModuleList([
                Upsample1D(curr_dim),
                # Input to ResBlock is (Upsampled + Skip)
                ResidualBlock1D(curr_dim + out_dim, out_dim, time_dim),
                ResidualBlock1D(out_dim, out_dim, time_dim),
            ]))
            curr_dim = out_dim

        # 6. Output
        self.final_norm = nn.GroupNorm(8, curr_dim)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv1d(curr_dim, input_dim, 1)

    def forward(self, x, t, y=None):
        """
        x: Noisy Latent [B, D, T]  (D=128 or 1024)
        t: Timestep [B]
        y: Condition [B, D, T]
        """
        if y is None:
            raise ValueError("Condition y is required for generic Latent Drifting")
        
        condition = y
        
        # Time Embedding
        t_emb = get_timestep_embedding(t, self.model_dim)
        t_emb = self.time_mlp(t_emb)

        # Concat inputs (Conditioning)
        h = torch.cat([x, condition], dim=1)
        h = self.input_proj(h)
        
        # Store for skip connections
        skips = []
        
        # DOWN
        for res1, res2, down in self.downs:
            h = res1(h, t_emb)
            h = res2(h, t_emb)
            skips.append(h)
            h = down(h)

        # MID
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # UP
        for up, res1, res2 in self.ups:
            h = up(h)
            
            # Skip connection
            skip = skips.pop()
            
            # Handle shape mismatch due to padding (if T is odd)
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode='nearest')
            
            h = torch.cat([h, skip], dim=1)
            h = res1(h, t_emb)
            h = res2(h, t_emb)

        # FINAL
        h = self.final_act(self.final_norm(h))
        out = self.final_conv(h)
        return out
