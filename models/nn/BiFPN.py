import torch
import torch.nn as nn
 
__all__ = ['BiFPN_Concat']

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=1, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        return self.act(self.conv(x))

# --- 新增 SeparableConv 模組 ---
class SeparableConv(nn.Module):
    """Depthwise separable convolution."""
    def __init__(self, c1, c2, k=3, s=1, act=True):
        """
        Initializes a depthwise separable convolution block using the provided Conv module.
        c1: input channels
        c2: output channels
        k: kernel size for depthwise conv
        s: stride for depthwise conv
        """
        super().__init__()
        # 使用您定義的 Conv 模組來建構
        # 1. 深度卷積 (Depthwise): c1 -> c1, groups=c1
        self.conv_dw = Conv(c1, c1, k, s, p=None, g=1, act=act)
        # 2. 逐點卷積 (Pointwise): c1 -> c2
        self.conv_pw = Conv(c1, c2, 1, 1, p=None, g=1, act=act)

    def forward(self, x):
        return self.conv_pw(self.conv_dw(x))

    def forward_fuse(self, x):
        return self.conv_pw.forward_fuse(self.conv_dw.forward_fuse(x))
# --- 結束新增 ---


class BiFPN_Concat(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Concat, self).__init__()
        self.w1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        # 使用 1e-4 是 EfficientDet 論文中的標準值
        self.epsilon = 1e-4  # <--- MODIFIED (0.0001 也可以)
        
        # MODIFIED: 改用 SeparableConv (深度可分離卷積)
        self.conv = SeparableConv(c1, c2)
        
        # REMOVED: 移除額外的激活函數
        # self.act = nn.ReLU() <--- REMOVED
 
    def forward(self, x):  
        if len(x) == 2:
            # MODIFIED: 應用 ReLU 確保權重為非負數
            w = torch.relu(self.w1_weight)
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            
            # MODIFIED: 移除 self.act()
            x = self.conv(weight[0] * x[0] + weight[1] * x[1])
        
        elif len(x) == 3:
            # MODIFIED: 應用 ReLU 確保權重為非負數
            w = torch.relu(self.w2_weight)
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            
            # MODIFIED: 移除 self.act()
            x = self.conv(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2])
        
        return x