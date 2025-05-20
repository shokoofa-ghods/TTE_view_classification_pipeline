"""
Module containing Video model
"""

import torch
import torch.nn as nn
import torchvision

from utils import VALID_LABELS

CNN_OUTPUT_SIZE = 1000
NUM_CLASSES = len(VALID_LABELS)
HIDDEN_SIZE = 1000

class CNNLSTM(nn.Module):
    """
    Video CNNLSTM Model
    """
    def __init__(self, num_layers:int=2):
        super(CNNLSTM, self).__init__()
        self.num_layers = num_layers

        self.conv = torchvision.models.efficientnet_b2(weights='DEFAULT')
        # for module in self.conv.features:
        #     if isinstance(module, torch.nn.modules.container.Sequential):
        #         module.append(torch.nn.Dropout(0.4))
        self.lstm = nn.LSTM(CNN_OUTPUT_SIZE,
                            HIDDEN_SIZE,
                            self.num_layers,
                            batch_first=True,
                            dropout = 0.1)
        self.attention_layer = nn.Linear(HIDDEN_SIZE, 1)
        self.classifier_layer = nn.Sequential(nn.Dropout(0.2),
                                              nn.Linear(HIDDEN_SIZE, NUM_CLASSES))

    def attention_net(self, lstm_out:torch.Tensor) -> torch.Tensor:
        """
        Apply the attention layer to the LSTM output

        Args:
            lstm_out (Tensor): output of the lstm block
        
        Returns:
            The Tensor output of the attention layer
        """
        attention_weights = torch.tanh(self.attention_layer(lstm_out))
        attention_weights = torch.softmax(attention_weights, dim=1)
        attention_out = torch.bmm(attention_weights.transpose(1, 2), lstm_out)
        return attention_out.squeeze(1)

    def forward(self, x):
        """
        Forward pass
        """
        batch_size, seq_len, c, h, w = x.size()
        c_in = x.view(batch_size * seq_len, c, h, w)
        # print(self.conv)
        c_out = self.conv(c_in)
        lstm_in = c_out.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(lstm_in)

        lstm_out = lstm_out[:, -1, :]
        output = self.classifier_layer(lstm_out)

        # attention_out = self.attention_net(lstm_out)
        # output = self.classifier_layer(attention_out)

        return output, torch.softmax(output, dim = 1)
