import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=32
        )
        self.encoder_output_layer = nn.Linear(
            in_features=32, out_features=kwargs["features"]
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=kwargs["features"], out_features=32
        )
        self.decoder_output_layer = nn.Linear(
            in_features=32, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed

    def feature_extraction(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        return code

    
