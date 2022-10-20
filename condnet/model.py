from typing import List, Dict

import torch
import torch.nn as nn
from segmentation_models.condnet.module import EncMod, DecMod, CnvMod, Map 

from typing import List, Dict

class EncoderTrack(nn.Module):
    def __init__(self):
        super(EncoderTrack, self).__init__()
        self.encodermodules = nn.ModuleList([
            EncMod(1 if i == 0 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(0, 5)
        ])
        self.decodermodule = DecMod(64, 64)
        self.encodermodule = EncMod(64, 128)

    def encoder_track(self, x):
        for encoder in self.encodermodules:
            x = encoder(x)
        return self.encodermodule(self.decodermodule(x))
    
    def forward(self, x):
        return torch.concat((self.encoder_track(x), self.encoder_track(x)))

class EncoderSubTrack(nn.Module):
    def __init__(self, num):
        super(EncoderSubTrack, self).__init__()
        self.encodermodules = nn.ModuleList([
            EncMod(1 if i == 0 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(0, num) 
        ])
    def encoder_track(self, x):
        for encoder in self.encodermodules:
            x = encoder(x)
        return x
    def forward(self, x):
        return torch.concat((self.encoder_track(x), self.encoder_track(x)))

class DecoderTrack(nn.Module):
    def __init__(self):
        super(DecoderTrack, self).__init__()
        self.convmodules = nn.ModuleList([ 
            CnvMod(128 if i == 5 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(5, 0, -1)
            ])
        self.decodermodules = nn.ModuleList([ 
            DecMod(2 ** (i + 2), 2 ** (i + 1)) for i in range(5, 0, -1)
            ])
        self.map = nn.ModuleList([ 
            Map(2 ** (i + 1), 2 ** (i)) for i in range(5, 0, -1)
            ])
        self.sub_encoder = nn.ModuleList([
            EncoderSubTrack(i) for i in range(4, 0, -1)
            ])

    def forward(self, out, x):
        for cnv, dec, mp, sub in zip(self.convmodules, self.decodermodules, self.map, self.sub_encoder):
            out = mp(dec(cnv(out)))
            res = sub(x)
            out = torch.concat((out, res[:, :, : out.shape[2], : out.shape[2]]))
        return out
        
class CondNet(nn.Module):
    def __init__(self, p = 0.5):
        super(CondNet, self).__init__()
        self.encoder = EncoderTrack()
        self.decoder = DecoderTrack()
        self.dropout = nn.Dropout(p)

    def forward(self, x, dropout_train = False):
        if dropout_train == False:
            out = self.encoder(x)
            out = self.decoder(out, x)            
        elif dropout_train == True:
            out = self.dropout(self.encoder(x))
            out = self.dropout(self.decoder(out, x))
        return out[0:10, 0:1, :, :]
        
