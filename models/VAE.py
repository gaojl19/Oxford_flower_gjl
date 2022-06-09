import jittor as jt
from jittor import nn
from functools import partial

from .model import *
from .transformer import *

MIDDLE_DIM = 1024
OUTPUT_DIM = 224*224

### CNN
class ConvMixer_vae(nn.Module):
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

        # add g: 2-layer mlp with relu
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, OUTPUT_DIM)
        )
    
    def execute(self, x):
        embedding = self.embedding(x)
        embedding = self.blocks(embedding)
        print(embedding.shape)
        embedding = self.avgpool(embedding)     # feature vector!!!!
        embedding = jt.flatten(embedding, 1)
        print(embedding.shape)
        exit(0)
        out = self.decoder(embedding)
        return out

    def feature(self, x):
        embedding = self.embedding(x)
        embedding = self.blocks(embedding)
        embedding = self.avgpool(embedding)     # feature vector!!!!
        embedding = jt.flatten(embedding, 1)
        return embedding

def ConvMixer_768_32_vae(num_classes: int = 1000, **kwargs):
    model = ConvMixer_vae(dim = 768, depth = 32, 
                    kernel_size=7, patch_size=7,
                    n_classes=num_classes,
                    **kwargs)
    return model



### MLP MIXER
class MLPMixer_vae(MLPMixer):
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

        # self.active = nn.LayerNorm(d_model)
        # self.mlp_head = nn.Sequential(
        #     nn.Linear(d_model, num_classes)
        # )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, OUTPUT_DIM)
        )
        

    def execute(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)     # feature vector!!!!
        # embedding = self.active(embedding)
        print(embedding.shape)
        embedding = embedding.mean(dim=1)
        print(embedding.shape)
        exit(0)
        out = self.decoder(embedding)
        # out = self.mlp_head(embedding)
        return out

    def feature(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)     # feature vector!!!!
        # embedding = self.active(embedding)
        embedding = embedding.mean(dim=1)
        return embedding

def MLPMixer_S_16_vae(num_classes: int = 1000, **kwargs):

    model = MLPMixer_vae(patch_size = 16, d_model = 512, depth = 12, 
                    num_classes=num_classes,
                    **kwargs)
    return model



# TRANSFORMER
class VisionTransformer_vae(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 hybrid_backbone=None, 
                 norm_layer=nn.LayerNorm):
        super(VisionTransformer_vae,self).__init__()
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches

        self.cls_token = jt.zeros((1, 1, embed_dim))
        self.pos_embed = jt.zeros((1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        # self.head = nn.Linear(embed_dim, num_classes)

        # add decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, OUTPUT_DIM)
        )

        self.pos_embed = trunc_normal_(self.pos_embed, std=.02)
        self.cls_token = trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
    
    def apply(self,fn):
        for m in self.modules():
            fn(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def execute(self, x, return_feature=False, return_attn=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        attention_weights = []
        
        _,i,j = self.cls_token.shape
        cls_tokens = self.cls_token.expand((B, i, j))  # stole cls_tokens impl from Phil Wang, thanks

        x = jt.contrib.concat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            if return_attn:
                x, attn_weights = blk(x, return_attn)
                attention_weights.append(attn_weights)
            else:
                x = blk(x)
                                        
        x = self.norm(x)
        feature_vector = x  # feature vector!!!!
        # x = self.head(x[:, 0])
        print(x.shape)
        print(x[:, 0].shape)
        exit(0)
        x = self.decoder(x[:, 0])

        if return_feature:
            if return_attn:
                return x, feature_vector, attention_weights
            else:
                return x, feature_vector
        else:
            if return_attn:
                return x, attention_weights
        return x

    def feature(self, x, return_feature=False, return_attn=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        attention_weights = []

        _, i, j = self.cls_token.shape
        cls_tokens = self.cls_token.expand((B, i, j))  # stole cls_tokens impl from Phil Wang, thanks

        x = jt.contrib.concat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            if return_attn:
                x, attn_weights = blk(x, return_attn)
                attention_weights.append(attn_weights)
            else:
                x = blk(x)

        x = self.norm(x)
        feature_vector = x  # feature vector!!!!
        # print("FEATURE SHAPE", feature_vector.shape)
        # x = self.head(x[:, 0])
        return feature_vector


def vit_small_patch16_224_vae(pretrained=False, **kwargs):
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer_vae(patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['vit_small_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=kwargs.get('num_classes', 0), in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model