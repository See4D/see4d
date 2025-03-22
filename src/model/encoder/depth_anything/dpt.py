import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from .blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        # if nclass > 1:
        #     self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        #     self.scratch.output_conv = nn.Sequential(
        #         nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(True),
        #         nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
        #     )
        # else:
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, nclass, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if nclass == 1 else nn.Identity(),
            # nn.Identity(),
        )
            
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out
        
        
class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super(DPT_DINOv2, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl']
        

        # self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
        # in case the Internet connection is not stable, please load the DINOv2 locally
        if localhub:
            self.pretrained = torch.hub.load('src/model/encoder/depth_anything/torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
            
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x):
        h, w = x.shape[-2:]
        
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return {"depth":depth.squeeze(1),
                "features": features}


class DepthAnything(DPT_DINOv2, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(**config)

# Identify mismatched layers
def filter_mismatched_layers(pretrained_state_dict, model_state_dict):
    filtered_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k in model_state_dict and v.size() == model_state_dict[k].size():
            filtered_state_dict[k] = v
        else:
            print(f"Skipping mismatched layer: {k} with shape {v.size()} vs {model_state_dict[k].size()}")
    return filtered_state_dict


class DepthAnythingGaussians(DPT_DINOv2, PyTorchModelHubMixin):
    def __init__(self, out_dim, gaussians_per_pixel=1, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super().__init__(encoder, features, out_channels, use_bn, use_clstoken, localhub)
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.depth_head = DPTHead(out_dim, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        
        self.depth_activation = nn.Sigmoid()
        self.opacities_activation = nn.Sigmoid()
        self.gaussians_per_pixel = gaussians_per_pixel
        
    def forward(self, x):
        h, w = x.shape[-2:]
        
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        
        patch_h, patch_w = h // 14, w // 14

        output = self.depth_head(features, patch_h, patch_w)
        B, _ ,H,W = output.size()
        # import ipdb; ipdb.set_trace()
        raw_gaussians = output.reshape(B, -1, self.gaussians_per_pixel, H, W) # B,C,N,H,W
        # the first channel of output is depth
        depth = raw_gaussians[:, 0] # B,N,H,W
        # B,N,H,W = depth.size()
        # the second channel in raw gaussians is opacity
        opacities = raw_gaussians[:, 1:2] # B,1,N,H,W
        ## othersare gaussian attributes
        raw_gaussians = raw_gaussians[:, 2:] # B,C,N,H,W
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        # depth = F.relu(depth)

        return {"depths": self.depth_activation(depth), # B,N,H,W
                "opacities": self.opacities_activation(opacities), # B,1,N,H,W
                "raw_gaussians": raw_gaussians, # B,C,N,H,W
                }

    def load_from_pretrained(self, pretrained_model_name_or_path: str, **kwargs):
        # import ipdb; ipdb.set_trace()
        if "-hf" in pretrained_model_name_or_path:
            # load model from huggingface hub
            missing_keys, unexpected_keys = self.load_state_dict(torch.load(hf_hub_download(pretrained_model_name_or_path, **kwargs)), strict=False)
        else:
            ## load model from pytorch hub
            # import ipdb; ipdb.set_trace()
            model = DepthAnything.from_pretrained(pretrained_model_name_or_path)
            # Filter the mismatched layers
            filtered_state_dict = filter_mismatched_layers(model.state_dict(), self.state_dict())
            missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
            
        print("Missing keys: ", missing_keys)
        print("Unexpected keys: ", unexpected_keys)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        default="vits",
        type=str,
        choices=["vits", "vitb", "vitl"],
    )
    args = parser.parse_args()
    
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_{:}14".format(args.encoder))
    
    print(model)
    