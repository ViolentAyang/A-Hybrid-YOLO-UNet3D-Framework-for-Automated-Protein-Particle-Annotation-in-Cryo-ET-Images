import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import cc3d
import copick
from typing import List, Tuple, Union
from tqdm import tqdm
from monai.networks.nets import UNet
from monai.networks.blocks import ResidualUnit
from monai.data import DataLoader, Dataset, CacheDataset
from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    Orientationd,  
    NormalizeIntensityd,
)

from config import (
    TRAIN_DATA_DIR, 
    copick_user_name, 
    copick_segmentation_name,
    voxel_size,
    tomo_type,
    id_to_name,
    BLOB_THRESHOLD,
    CERTAINTY_THRESHOLD,
    classes
)
from data_utils import extract_3d_patches_minimal_overlap, reconstruct_array, dict_to_df

class ResUnit3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class CustomUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=7, channels=(48, 64, 80, 80), strides=(2, 2, 1)):
        super().__init__()
        
        # Encoder
        self.enc1 = ResUnit3D(in_channels, channels[0])
        self.enc2 = ResUnit3D(channels[0], channels[1], stride=strides[0])
        self.enc3 = ResUnit3D(channels[1], channels[2], stride=strides[1])
        self.enc4 = ResUnit3D(channels[2], channels[3], stride=strides[2])
        
        # Bottleneck
        self.bottleneck = ResUnit3D(channels[3], channels[3])
        
        # Channel attention for skip connections
        self.att1 = ChannelAttention3D(channels[0])
        self.att2 = ChannelAttention3D(channels[1])
        self.att3 = ChannelAttention3D(channels[2])
        
        # Skip connection adjustment convolutions
        self.skip_conv1 = nn.Conv3d(channels[0], channels[0], 1, 1)
        self.skip_conv2 = nn.Conv3d(channels[1], channels[1], 1, 1)
        self.skip_conv3 = nn.Conv3d(channels[2], channels[2], 1, 1)
        
        # Decoder with transposed convolutions
        self.up3 = nn.ConvTranspose3d(channels[3], channels[2], kernel_size=2, stride=strides[2] if strides[2] > 1 else 1)
        self.dec3 = ResUnit3D(channels[2] * 2, channels[2])
        
        self.up2 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=2, stride=strides[1])
        self.dec2 = ResUnit3D(channels[1] * 2, channels[1])
        
        self.up1 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=2, stride=strides[0])
        self.dec1 = ResUnit3D(channels[0] * 2, channels[0])
        
        # Final output layer
        self.final_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder path with skip connections and attention
        d3 = self.up3(b)
        skip3 = self.att3(e3) * self.skip_conv3(e3)
        if d3.shape != skip3.shape:
            d3 = F.interpolate(d3, size=skip3.shape[2:], mode='trilinear', align_corners=False)
        d3 = torch.cat([d3, skip3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        skip2 = self.att2(e2) * self.skip_conv2(e2)
        if d2.shape != skip2.shape:
            d2 = F.interpolate(d2, size=skip2.shape[2:], mode='trilinear', align_corners=False)
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        skip1 = self.att1(e1) * self.skip_conv1(e1)
        if d1.shape != skip1.shape:
            d1 = F.interpolate(d1, size=skip1.shape[2:], mode='trilinear', align_corners=False)
        d1 = torch.cat([d1, skip1], dim=1)
        d1 = self.dec1(d1)
        
        output = self.final_conv(d1)
        return output

class Model(pl.LightningModule):
    def __init__(
        self, 
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 7,
        channels: Union[Tuple[int, ...], List[int]] = (48, 64, 80, 80),
        strides: Union[Tuple[int, ...], List[int]] = (2, 2, 1),
        num_res_units: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = CustomUNet3D(
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            channels=self.hparams.channels,
            strides=self.hparams.strides,
        )
    
    def forward(self, x):
        return self.model(x)

def setup_copick_config():
    copick_config_path = TRAIN_DATA_DIR + "/copick.config"
    
    with open(copick_config_path) as f:
        copick_config = json.load(f)
    
    copick_config['static_root'] = './data/test/static'
    
    copick_test_config_path = 'copick_test.config'
    
    with open(copick_test_config_path, 'w') as outfile:
        json.dump(copick_config, outfile)
    
    return copick_test_config_path

def load_models(model_paths):
    models = []
    for model_path in model_paths:
        channels = (48, 64, 80, 80)
        strides_pattern = (2, 2, 1)       
        num_res_units = 1
        model = Model(channels=channels, strides=strides_pattern, num_res_units=num_res_units)
        
        weights = torch.load(model_path)['state_dict']
        model.load_state_dict(weights)
        model.to('cuda')
        model.eval()
        models.append(model)
    return models

def ensemble_prediction_tta(models, input_tensor, threshold=0.5):
    model = models[0]
    probs_list = []
    data_copy0 = input_tensor.clone()
    data_copy0 = torch.flip(data_copy0, dims=[2])
    data_copy1 = input_tensor.clone()
    data_copy1 = torch.flip(data_copy1, dims=[3])
    data_copy2 = input_tensor.clone()
    data_copy2 = torch.flip(data_copy2, dims=[4])
    
    with torch.no_grad():
        model_output0 = model(input_tensor)
        model_output1 = model(data_copy0)
        model_output1 = torch.flip(model_output1, dims=[2])
        model_output2 = model(data_copy1)
        model_output2 = torch.flip(model_output2, dims=[3])
        model_output3 = model(data_copy2)
        model_output3 = torch.flip(model_output3, dims=[4])
        probs0 = torch.softmax(model_output0[0], dim=0)
        probs1 = torch.softmax(model_output1[0], dim=0)
        probs2 = torch.softmax(model_output2[0], dim=0)
        probs3 = torch.softmax(model_output3[0], dim=0)
        probs_list.append(probs0)
        probs_list.append(probs1)
        probs_list.append(probs2)
        probs_list.append(probs3)
    
    avg_probs = torch.mean(torch.stack(probs_list), dim=0)
    thresh_probs = avg_probs > threshold
    _, max_classes = thresh_probs.max(dim=0)
    return max_classes

def run_unet_inference():
    copick_test_config_path = setup_copick_config()
    root = copick.from_file(copick_test_config_path)
    
    inference_transforms = Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image"], axcodes="RAS")
    ])
    
    model_paths = [
        './models/unet_custom_trained.ckpt',
    ]
    
    models = load_models(model_paths)
    
    location_df = []
    for run in root.runs:
        tomo = run.get_voxel_spacing(10)
        tomo = tomo.get_tomogram(tomo_type).numpy()
        tomo_patches, coordinates = extract_3d_patches_minimal_overlap([tomo], 96)
        tomo_patched_data = [{"image": img} for img in tomo_patches]
        tomo_ds = CacheDataset(data=tomo_patched_data, transform=inference_transforms, cache_rate=1.0)
        pred_masks = []
        for i in tqdm(range(len(tomo_ds))):
            input_tensor = tomo_ds[i]['image'].unsqueeze(0).to("cuda")
            max_classes = ensemble_prediction_tta(models, input_tensor, threshold=CERTAINTY_THRESHOLD)
            pred_masks.append(max_classes.cpu().numpy())
        reconstructed_mask = reconstruct_array(pred_masks, coordinates, tomo.shape)
        location = {}
        for c in classes:
            cc = cc3d.connected_components(reconstructed_mask == c)
            stats = cc3d.statistics(cc)
            zyx = stats['centroids'][1:] * 10.012444
            zyx_large = zyx[stats['voxel_counts'][1:] > BLOB_THRESHOLD]
            xyz = np.ascontiguousarray(zyx_large[:, ::-1])
            location[id_to_name[c]] = xyz
        df = dict_to_df(location, run.name)
        location_df.append(df)
    
    location_df = pd.concat(location_df)
    location_df.insert(loc=0, column='id', value=np.arange(len(location_df)))
    
    return location_df 