import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # -> (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # -> (B, num_patches, embed_dim)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        in_channels=5,
        output_channels=2,
        patch_size=4,
        embed_dim=128,
        depth=6,
        num_heads=8,
        img_size=(48, 72),
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, patch_size, embed_dim)
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # back to 2d
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, patch_size * patch_size * output_channels)
        )

        self.patch_size = patch_size
        self.img_size = img_size
        self.output_channels = output_channels

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        x = x + self.pos_embed[:, :x.size(1)]

        x = self.transformer(x)  # (B, num_patches + 1, embed_dim)

        # Project each patch to (patch_area * output_channels)
        x = self.head(x)  # (B, num_patches, patch_area * output_channels)

        # back to 2d
        num_patches_h = self.img_size[0] // self.patch_size
        num_patches_w = self.img_size[1] // self.patch_size
        x = x.view(B, num_patches_h, num_patches_w, self.patch_size, self.patch_size, self.output_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C_out, H, W)
        x = x.view(B, self.output_channels, self.img_size[0], self.img_size[1])
        return x