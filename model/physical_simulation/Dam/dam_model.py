from torch.nn.modules import Module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from base import *

DAM_settings = {"obs_dim": [2, 64, 64],
                    "state_dim": [2, 64, 64], 
                    "seq_length": 12,
                    "obs_feature_dim": [512, 128, 64, 32, 16, 8], 
                    "state_filter_feature_dim": [16, 32, 64, 128, 256]}

DAM_settings["state_feature_dim"] = [4096, 512]


'''
=========================
NN features for DAM Flow
=========================
'''


class DAM_ENCODER(Module):
    def __init__(self, *args, **kwargs) -> None:
        super(DAM_ENCODER, self).__init__(*args, **kwargs)
        self.input_dim, self.w, self.h = DAM_settings["state_dim"]
        self.filter_dims = DAM_settings["state_filter_feature_dim"]
        self.hidden_dims = DAM_settings["state_feature_dim"] # [Dim before linear, state_feature_dim]

        # First convolution layer with larger kernel for feature extraction
        self.Conv2D_size7_1 = nn.Conv2d(in_channels=self.input_dim, out_channels=self.filter_dims[0], 
                                  kernel_size=7, stride=1, padding=3)
        
        # Second convolution layer
        self.Conv2D_size5_1 =  nn.Conv2d(in_channels=self.filter_dims[0], out_channels=self.filter_dims[1], 
                                  kernel_size=5, stride=1, padding=2)
        
        # Third convolution layer
        self.Conv2D_size3_1 = nn.Conv2d(in_channels=self.filter_dims[1], out_channels=self.filter_dims[2], 
                                              kernel_size=3, stride=1, padding=1)
        
        # Fourth convolution layer
        self.Conv2D_size3_2 = nn.Conv2d(in_channels=self.filter_dims[2], out_channels=self.filter_dims[3], 
                                              kernel_size=3, stride=1, padding=1)
        
        # Fifth convolution layer
        self.Conv2D_size3_3 = nn.Conv2d(in_channels=self.filter_dims[3], out_channels=self.filter_dims[4], 
                                              kernel_size=3, stride=1, padding=1)

        self.flatten = nn.Flatten()
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
        self.linear = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])


    def forward(self, state: torch.Tensor):
        # First layer: 7x7 conv + pooling
        en_state_1 = self.Conv2D_size7_1(state)
        en_state_1 = self.pooling(en_state_1)
        en_state_1 = self.relu(en_state_1)

        # Second layer: 5x5 conv + pooling
        en_state_2 = self.Conv2D_size5_1(en_state_1)
        en_state_2 = self.relu(en_state_2)
        en_state_2 = self.pooling(en_state_2)

        # Third layer: 3x3 conv + pooling
        en_state_3 = self.Conv2D_size3_1(en_state_2)
        en_state_3 = self.pooling(en_state_3)
        en_state_3 = self.relu(en_state_3)

        # Fourth layer: 3x3 conv + pooling
        en_state_4 = self.Conv2D_size3_2(en_state_3)
        en_state_4 = self.pooling(en_state_4)
        en_state_4 = self.relu(en_state_4)

        # Fifth layer: 3x3 conv
        en_state_5 = self.Conv2D_size3_3(en_state_4)
        en_state_5 = self.relu(en_state_5)
        en_state_5 = self.dropout(en_state_5)

        # Flatten and linear transformation
        en_state_5 = self.flatten(en_state_5)
        z = self.linear(en_state_5)

        return z


class DAM_DECODER(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(DAM_DECODER, self).__init__(*args, **kwargs)
        self.input_dim, self.w, self.h = DAM_settings["state_dim"]
        self.filter_dims = DAM_settings["state_filter_feature_dim"]
        self.hidden_dims = DAM_settings["state_feature_dim"] # [Dim before linear, state_feature_dim]

        self.linear = nn.Linear(self.hidden_dims[1], self.hidden_dims[0])
        
        # Transpose convolution layers (reverse order of encoder)
        self.ConvTranspose2D_size3_1 = nn.ConvTranspose2d(in_channels=self.filter_dims[4], out_channels=self.filter_dims[3], 
                                                          kernel_size=3, stride=1, padding=1)
        self.ConvTranspose2D_size3_2 = nn.ConvTranspose2d(in_channels=self.filter_dims[3], out_channels=self.filter_dims[2],
                                                          kernel_size=3, stride=1, padding=1)
        self.ConvTranspose2D_size3_3 = nn.ConvTranspose2d(in_channels=self.filter_dims[2], out_channels=self.filter_dims[1],
                                                          kernel_size=3, stride=1, padding=1)
        
        self.Upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.ConvTranspose2D_size5_1 = nn.ConvTranspose2d(in_channels=self.filter_dims[1], out_channels=self.filter_dims[0],
                                                          kernel_size=5, stride=1, padding=2)
        
        self.ConvTranspose2D_size7_1 = nn.ConvTranspose2d(in_channels=self.filter_dims[0], out_channels=self.input_dim,
                                                          kernel_size=7, stride=1, padding=3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Output refinement layers
        self.output_conv = nn.Sequential(nn.Conv2d(in_channels=self.input_dim, out_channels=64, 
                                         kernel_size=1, stride=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=64, out_channels=32, 
                                         kernel_size=1, stride=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=32, out_channels=self.input_dim, 
                                         kernel_size=1, stride=1))


    def forward(self, z: torch.Tensor):
        # Linear transformation and reshape
        de_state_5 = self.linear(z)
        de_state_5 = self.relu(de_state_5)
        de_state_5 = self.dropout(de_state_5)
        de_state_5 = de_state_5.view(-1, self.filter_dims[4], 4, 4)  # Reshape to [batch, 256, 4, 4]

        # First transpose conv
        de_state_4 = self.ConvTranspose2D_size3_1(de_state_5)
        de_state_4 = self.relu(de_state_4)

        # Second transpose conv with upsampling
        de_state_3 = self.Upsampling(de_state_4)
        de_state_3 = self.ConvTranspose2D_size3_2(de_state_3)
        de_state_3 = self.relu(de_state_3)

        # Third transpose conv with upsampling
        de_state_2 = self.Upsampling(de_state_3) 
        de_state_2 = self.ConvTranspose2D_size3_3(de_state_2)
        de_state_2 = self.relu(de_state_2)

        # Fourth transpose conv with upsampling
        de_state_1 = self.Upsampling(de_state_2)
        de_state_1 = self.ConvTranspose2D_size5_1(de_state_1)
        de_state_1 = self.relu(de_state_1)

        # Final transpose conv with upsampling
        de_state_0 = self.Upsampling(de_state_1)
        de_state_0 = self.ConvTranspose2D_size7_1(de_state_0)
        
        # Output refinement
        recon_s = self.output_conv(de_state_0)

        return recon_s


'''
=======================
Operators for DAM Flow
=======================
'''


class DAM_ROM(ROM_BASE):
    def __init__(self, *args, **kwargs) -> None:
        ENCODER = DAM_ENCODER()
        DECODER = DAM_DECODER()
        seq_length = DAM_settings["seq_length"]
        super(DAM_ROM, self).__init__(ENCODER=ENCODER,
                                        DECODER=DECODER, 
                                        seq_length=seq_length,
                                        *args, **kwargs)


if __name__ == "__main__":
    model = DAM_ROM()
    print(f"Model created successfully!")
    print(f"State dimension: {DAM_settings['state_dim']}")
    print(f"Sequence length: {DAM_settings['seq_length']}")
    
    dummy_input = torch.randn(1, 2, 64, 64)
    with torch.no_grad():
        encoded = model.K_S(dummy_input)
        decoded = model.K_S_preimage(encoded)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Decoded shape: {decoded.shape}")
    print("Test completed successfully!")