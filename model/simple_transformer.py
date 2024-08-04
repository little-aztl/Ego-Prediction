import torch
import torch.nn as nn

class Simple_Transformer_Eye_Gaze(nn.Module):
    def __init__(self, d_model=512):
        super(Simple_Transformer_Eye_Gaze, self).__init__()
        self.transformer = nn.Transformer(d_model=3, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)

    def forward(self, ipt, tgt):
        return self.transformer(ipt, tgt)
        