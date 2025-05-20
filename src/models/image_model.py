"""
Module containing Image Model
"""

import torch.nn as nn
import torchvision
import torch

from utils import VALID_LABELS

NUM_CLASSES = len(VALID_LABELS)

class SEBlock(nn.Module):
    """
    Squeeze Excitation Block
    """
    def __init__(self, input_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_channels // reduction, input_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward Pass
        """
        batch_size, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        return x * y

class CNNLSTM(nn.Module):
    """
    Image CNN LSTM Model
    """
    def __init__(self):
        super(CNNLSTM, self).__init__()

        # Load EfficientNet B2 without the final classification layer
        self.conv = torchvision.models.efficientnet_b2(pretrained=True).features

        # EfficientNet B2 output feature map has 1408 channels
        cnn_output_channels = 1408

        # SE block after EfficientNet
        self.se_block = SEBlock(input_channels=cnn_output_channels)

        # Classifier layer
        self.classifier_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Pool to (1,1) size
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(cnn_output_channels, NUM_CLASSES)
        )

    def forward(self, x):
        """
        Forward pass
        """
        _batch_size, _frame1, _c, _h, _w = x.size()
        x = x.view(_batch_size * _frame1, _c, _h, _w)
        c_out = self.conv(x)
        c_out = self.se_block(c_out)  # Apply SE block
        output = self.classifier_layer(c_out)
        return output, torch.softmax(output, dim = 1)
