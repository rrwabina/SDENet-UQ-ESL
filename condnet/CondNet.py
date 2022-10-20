import torch
import torch.nn as nn
from typing import List, Dict


class CnvMod(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size = (5, 5)):
        super(CnvMod, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 
                        kernel_size = (1, 1), bias = False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace = True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class EncMod(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(EncMod, self).__init__()
        self.block = nn.Sequential(
            CnvMod(input_channel, output_channel, kernel_size = (3, 3)),
            nn.MaxPool2d(1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DecMod(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DecMod, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel,
                                kernel_size = (2, 2), bias = False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        return self.block(x)

class Map(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Map, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 
                        kernel_size = (4, 4)),
            nn.LogSigmoid()
        )
    def forward(self, x):
        return self.block(x)     

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
    
    def forward(self, a, b):
        return torch.concat((self.encoder_track(a), self.encoder_track(b)))

class EncoderSubTrackA(nn.Module):
    def __init__(self):
        super(EncoderSubTrackA, self).__init__()
        self.encodermodules = nn.ModuleList([
            EncMod(1 if i == 0 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(0, 1) 
        ])
    def encoder_track(self, x):
        for encoder in self.encodermodules:
            x = encoder(x)
        return x
    def forward(self, a, b):
        return torch.concat((self.encoder_track(a), self.encoder_track(b)))

class EncoderSubTrackB(nn.Module):
    def __init__(self):
        super(EncoderSubTrackB, self).__init__()
        self.encodermodules = nn.ModuleList([
            EncMod(1 if i == 0 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(0, 2) 
        ])
    def encoder_track(self, x):
        for encoder in self.encodermodules:
            x = encoder(x)
        return x
    def forward(self, a, b):
        return torch.concat((self.encoder_track(a), self.encoder_track(b)))

class EncoderSubTrackC(nn.Module):
    def __init__(self):
        super(EncoderSubTrackC, self).__init__()
        self.encodermodules = nn.ModuleList([
            EncMod(1 if i == 0 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(0, 3) 
        ])
    def encoder_track(self, x):
        for encoder in self.encodermodules:
            x = encoder(x)
        return x
    def forward(self, a, b):
        return torch.concat((self.encoder_track(a), self.encoder_track(b)))

class EncoderSubTrackD(nn.Module):
    def __init__(self):
        super(EncoderSubTrackD, self).__init__()
        self.encodermodules = nn.ModuleList([
            EncMod(1 if i == 0 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(0, 4) 
        ])
    def encoder_track(self, x):
        for encoder in self.encodermodules:
            x = encoder(x)
        return x
    def forward(self, a, b):
        return torch.concat((self.encoder_track(a), self.encoder_track(b)))

class DecoderTrack(nn.Module):
    def __init__(self):
        super(DecoderTrack, self).__init__()
        self.convmodules = nn.ModuleList([ 
            CnvMod(2 ** (i + 2), 2 ** (i + 2)) for i in range(5, 0, -1)
            ])
        self.decodermodules = nn.ModuleList([ 
            DecMod(2 ** (i + 2), 2 ** (i + 1)) for i in range(5, 0, -1)
            ])
        self.map = nn.ModuleList([ 
            Map(2 ** (i + 1), 2 ** (i + 2)) for i in range(5, 0, -1)
            ])

    def forward(self, x):
        for cnv, dec, mp in zip(self.convmodules, self.decodermodules, self.map):
            x = mp(dec(cnv(x)))
        return x

class DecoderTrackA(nn.Module):
    def __init__(self):
        super(DecoderTrackA, self).__init__()
        self.convmodules = CnvMod(128, 128)
        self.decodermodules = DecMod(128, 64)
        self.map = Map(64, 32)

    def forward(self, x):
        return self.map(self.decodermodules(self.convmodules(x)))

class DecoderTrackB(nn.Module):
    def __init__(self):
        super(DecoderTrackB, self).__init__()
        self.convmodules = CnvMod(32, 64)
        self.decodermodules = DecMod(64, 32)
        self.map = Map(32, 16)

    def forward(self, x):
        return self.map(self.decodermodules(self.convmodules(x)))

class DecoderTrackC(nn.Module):
    def __init__(self):
        super(DecoderTrackC, self).__init__()
        self.convmodules = CnvMod(16, 32)
        self.decodermodules = DecMod(32, 16)
        self.map = Map(16, 8)

    def forward(self, x):
        return self.map(self.decodermodules(self.convmodules(x)))

class DecoderTrackD(nn.Module):
    def __init__(self):
        super(DecoderTrackD, self).__init__()
        self.convmodules = CnvMod(8, 16)
        self.decodermodules = DecMod(16, 8)
        self.map = Map(8, 4)

    def forward(self, x):
        return self.map(self.decodermodules(self.convmodules(x)))

class DecoderTrackE(nn.Module):
    def __init__(self):
        super(DecoderTrackE, self).__init__()
        self.convmodules = CnvMod(4, 8)
        self.decodermodules = DecMod(8, 4)
        self.map = Map(4, 2)

    def forward(self, x):
        return self.map(self.decodermodules(self.convmodules(x)))

class CondSkip(nn.Module):
    def __init__(self):
        super(CondSkip, self).__init__()
        self.sub_encoderD = EncoderSubTrackD()
        self.sub_encoderC = EncoderSubTrackC()
        self.sub_encoderB = EncoderSubTrackB()
        self.sub_encoderA = EncoderSubTrackA()

    def forward(self, a, b):
        skipA = self.sub_encoderD(a, b)      
        skipB = self.sub_encoderC(a, b)     
        skipC = self.sub_encoderB(a, b)      
        skipD = self.sub_encoderA(a, b)        
        return skipA, skipB, skipC, skipD

class ConductorNetwork(nn.Module):
    def __init__(self):
        super(ConductorNetwork, self).__init__()
        self.encoder = EncoderTrack()

        self.decoderA = DecoderTrackA()
        self.decoderB = DecoderTrackB()
        self.decoderC = DecoderTrackC()
        self.decoderD = DecoderTrackD()
        self.decoderE = DecoderTrackE()

        self.skipencoders = CondSkip()

    def forward(self, a, b):
        skipA, skipB, skipC, skipD = self.skipencoders(a, b)

        x = self.encoder(a, b)
        
        # x = self.decoderA(x)                                                    
        # x = torch.concat((x, skipA[:, :, :x.shape[2], :x.shape[2]]))    
        # x = self.decoderB(x)                                                    
        # x = torch.concat((x, skipB[:, :, :x.shape[2], :x.shape[2]]))            
        # x = self.decoderC(x)                                                    
        # x = torch.concat((x, skipC[:, :, :x.shape[2], :x.shape[2]]))            
        # x = self.decoderD(x)                                                    
        # x = torch.concat((x, skipD[:, :, :x.shape[2], :x.shape[2]]))             
        # x = self.decoderE(x)                                                    
 
        return x
        