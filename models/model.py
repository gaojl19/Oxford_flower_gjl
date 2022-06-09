
import jittor as jt
from jittor import nn
from functools import partial

### CNN

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def execute(self, x):
        return self.fn(x) + x

class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, padding=patch_size//2),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=(int)((kernel_size - 1) // 2)),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)],
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
    
    def execute(self, x):
        embedding = self.embedding(x)
        embedding = self.blocks(embedding)
        embedding = self.avgpool(embedding)     # feature vector!!!!
        embedding = jt.flatten(embedding, 1)
        out = self.classifier(embedding)
        return out

def ConvMixer_768_32(num_classes: int = 1000, **kwargs):
    model = ConvMixer(dim = 768, depth = 32, 
                    kernel_size=7, patch_size=7,
                    n_classes=num_classes,
                    **kwargs)
    return model

### MLP MIXER

def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

def check_sizes(image_size, patch_size):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)
    assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0, 'image height and width must be divisible by patch size'
    num_patches = (image_height // patch_height) * (image_width // patch_width)
    return num_patches

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def execute(self, x):
        return self.fn(self.norm(x)) + x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., dense = nn.Linear):
        super().__init__()
        self.net = nn.Sequential(
            dense(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def execute(self, x):
        return self.net(x)


class MLPMixer(nn.Module):
    def __init__(self, num_patches, d_model, depth, expansion_factor = 4, dropout = 0.):
        super().__init__()

        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model, FeedForward(num_patches, num_patches * expansion_factor, dropout, chan_first)),
                PreNormResidual(d_model, FeedForward(d_model, d_model * expansion_factor, dropout, chan_last))
            ) for _ in range(depth)]
        )

    def execute(self, x):
        return self.model(x)

class MLPMixerForImageClassification(MLPMixer):
    def __init__(
        self, 
        in_channels = 3, 
        d_model = 512, 
        num_classes = 1000, 
        patch_size = 16, 
        image_size = 224, 
        depth = 12, 
        expansion_factor = 4):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__(num_patches, d_model, expansion_factor, depth)

        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size),
        )

        self.active = nn.LayerNorm(d_model)
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def execute(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)     # feature vector!!!!
        embedding = self.active(embedding)
        embedding = embedding.mean(dim=1)
        out = self.mlp_head(embedding)
        return out

def MLPMixer_S_16(num_classes: int = 1000, **kwargs):

    model = MLPMixerForImageClassification(patch_size = 16, d_model = 512, depth = 12, 
                    num_classes=num_classes,
                    **kwargs)
    return model


### TRANSFORMER
### 这玩意我没看懂，所以单独放在别的文件夹里了
