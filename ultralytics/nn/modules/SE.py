import torch
import torch.nn as nn

__all__ = ['SEModule']

class SEModule(nn.Module):
    """
    獨立的 Squeeze-and-Excitation (SE) 模塊。
    
    可插入到 YOLO .yaml 檔案中的任何位置，
    用於對其「上一層」的特徵圖進行通道注意力重標定。
    """
    def __init__(self, c1, r=16):
        """
        初始化 SE 模塊。
        
        Args:
            c1 (int): 輸入通道數 (由 YOLO 解析器自動從上一層獲取)。
            r (int): 縮減率 (Reduction ratio)，用於瓶頸層 (預設為 16)。
        """
        super().__init__()
        
        # Squeeze (壓縮)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全域平均池化
        
        # Excitation (激勵)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // r, bias=False),  # 降維
            nn.ReLU(inplace=True),
            nn.Linear(c1 // r, c1, bias=False),  # 升維
            nn.Sigmoid()                        # 歸一化為 (0, 1) 的權重
        )

    def forward(self, x):
        """ 
        前向傳播：
        1. 獲取原始特徵圖 x
        2. Squeeze: 計算每個通道的平均值
        3. Excitation: 計算通道權重
        4. Rescale: 將權重乘回原始特徵圖 x
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)     # Squeeze
        y = self.fc(y).view(b, c, 1, 1)     # Excitation
        return x * y.expand_as(x)           # Rescale