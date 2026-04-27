import torch
import torch.nn as nn
import math

class SpaceToDepth(nn.Module):
    """
    自定義的 SpaceToDepth 模組 (PyTorch 1.7.0+ 中有 PixelUnshuffle)。
    這是一個無參數的操作，用於將 H/W 維度的資訊轉移到 C 通道。
    """
    def __init__(self, block_size=2):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        """
        Input: [B, C, H, W]
        Output: [B, C * (bs^2), H/bs, W/bs]
        """
        b, c, h, w = x.shape
        # 1. Reshape to: [B, C, H/bs, bs, W/bs, bs]
        x = x.reshape(b, c, h // self.bs, self.bs, w // self.bs, self.bs)
        # 2. Permute to: [B, bs, bs, C, H/bs, W/bs]
        x = x.permute(0, 3, 5, 1, 2, 4)
        # 3. Reshape to: [B, C*(bs^2), H/bs, W/bs]
        x = x.reshape(b, c * (self.bs**2), h // self.bs, w // self.bs)
        return x

class SEModule(nn.Module):
    """
    Ultralytics 標準的 Squeeze-and-Excite (SE) 模組。
    它會對輸入的通道 (c1) 應用注意力。
    """
    def __init__(self, c1, r=8):
        """
        初始化 SE 模組。
        Args:
            c1 (int): 輸入通道數。
            r (int): 瓶頸層的壓縮率 (reduction ratio)。
        """
        super().__init__()
        c2 = int(c1 / r)  # 瓶頸層的通道數
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.l1 = nn.Linear(c1, c2, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c2, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """ 前向傳播 """
        b, c, _, _ = x.shape
        # Squeeze
        y = self.avgpool(x).view(b, c)
        # Excite
        y = self.l1(y)
        y = self.act(y)
        y = self.l2(y)
        y = self.sig(y)
        # Reshape and Apply
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

class FocusSE(nn.Module):
    """
    FocusSE 模組 (SpaceToDepth + SEModule + PWConv)
    
    實現您的想法：
    1. SpaceToDepth: 將精細的空間資訊 "折疊" 到通道維度。
       [B, c1, 320, 320] -> [B, c1*4, 160, 160]
       
    2. SEModule: 對這些 "攜帶空間資訊的通道" 學習重要性權重。
       [B, c1*4, 160, 160] -> [B, c1*4, 160, 160] (已加權)
       
    3. PWConv (1x1): 融合加權後的特徵並降維到目標通道數 c2。
       [B, c1*4, 160, 160] -> [B, c2, 160, 160]
    """
    def __init__(self, c1, c2):
        """
        初始化 FocusSE 模組。
        
        Args:
            c1 (int): 輸入通道數 (例如 Layer 0 的 32)
            c2 (int): 最終輸出的通道數 (例如您期望的 64)
        """
        super().__init__()
        self.spd = SpaceToDepth(block_size=2)
        
        # SpaceToDepth 會將通道數 c1 變為 c1 * 4
        c_spd = c1 * 4
        
        # 整合 SE 模組
        self.se = SEModule(c_spd) # 對 SPD 輸出的 128 個通道應用 SE
        
        # 使用 1x1 卷積 (PWConv) 來調整通道數並融合特徵
        self.conv = nn.Conv2d(c_spd, c2, 1, 1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()  # YOLOv8/v5 標準激活函數

    def forward(self, x):
        """ 前向傳播 """
        x = self.spd(x)    # 1. SpaceToDepth
        x = self.se(x)     # 2. SEModule (新加入)
        x = self.conv(x)   # 3. PWConv
        x = self.bn(x)
        x = self.act(x)
        return x

