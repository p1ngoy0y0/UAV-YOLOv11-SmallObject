# ---------------------------------------------------------------------
# --- 開始：你客製化的 RFB_CBAM 模組 (已修復) ---
# ---------------------------------------------------------------------

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, c1, r=1):
        super(ChannelAttention, self).__init__()
        c_hidden = max(1, c1 // r)
        
        # 1. 使用 Linear (符合你的權重檔需求)
        self.mlp = nn.Sequential(
            nn.Linear(c1, c_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_hidden, c1, bias=False)   
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # [⚠️ 重要修正] Linear 層必須配合 view 使用
        # 1. 攤平: (B, C, 1, 1) -> (B, C)
        avg_in = self.avg_pool(x).view(b, c)
        max_in = self.max_pool(x).view(b, c)
        
        # 2. 進 MLP
        avg_out = self.mlp(avg_in)
        max_out = self.mlp(max_in)
        
        # 3. 還原: (B, C) -> (B, C, 1, 1) 以便廣播相乘
        avg_out = avg_out.view(b, c, 1, 1)
        max_out = max_out.view(b, c, 1, 1)
        
        return self.sigmoid(avg_out + max_out)


class RFB_CBAM(nn.Module):
    """
    RFB-CBAM (Standard Channel-First Version) with Global Residual
    
    Flow:
      1. Input
      2. Attention Branch: Input -> Channel Attn (Linear) -> Spatial Attn (RFB+SimAM)
      3. Output = Input + Attention_Result (Global Residual Add)
    """
    def __init__(self, c1, *args):
        super(RFB_CBAM, self).__init__()
        
        # --- 參數解析 ---
        if len(args) >= 1: c2 = args[0]
        else: c2 = c1
        if len(args) >= 2: r = args[1]
        else: r = 2
        
        # 殘差連結要求輸入輸出通道必須一致
        if c1 != c2:
            raise ValueError(f"RFB_CBAM (Res) 輸入(c1={c1})和輸出(c2={c2})通道數必須相等。")
        
        # 1. 通道注意力 (Linear Version)
        self.channel_attention = ChannelAttention(c1, r)
        
        # 2. 空間注意力 (RFB-SAM)
        self.branch1_k1 = nn.Conv2d(2, 1, 1)
        self.branch2_k2 = nn.Conv2d(2, 1, 3, padding=1)
        self.branch3_k3 = nn.Conv2d(2, 1, 5, padding=2)
        self.reduce_conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        # [A] 殘差備份 (Global Residual)
        residual = x

        # ==================================================
        # [B] 步驟 1: 通道優先 (Channel First)
        # ==================================================
        channel_weights = self.channel_attention(x)
        x_cam = x * channel_weights
        
        # ==================================================
        # [C] 步驟 2: 空間後手 (Spatial Second)
        # ==================================================
        # 空間注意力基於「已被通道篩選過」的特徵 (x_cam) 來計算
        avg_pool = torch.mean(x_cam, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_cam, dim=1, keepdim=True)
        x_pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        s_merged = torch.cat([
            self.branch1_k1(x_pooled),
            self.branch2_k2(x_pooled),
            self.branch3_k3(x_pooled)
        ], dim=1)
        s_reduced = self.reduce_conv(s_merged)    
        
        # 產生空間權重
        spatial_weights = torch.sigmoid(s_reduced)
        
        # 計算最終的注意力特徵 (Attention Branch Output)
        # 注意：這裡是將空間權重應用在 x_cam 上
        attention_out = x_cam * spatial_weights.expand_as(x)

        # ==================================================
        # [D] 步驟 3: 全域殘差連結
        # ==================================================
        # Output = Original + (Original * ChannelW * SpatialW)
        output = residual + attention_out
        
        return output


class RFB_CBAM_Spatial(nn.Module):
    """
    RFB-CBAM (Spatial-First Version) with Residual Connection
    
    Flow: 
      1. Input
      2. Attention Branch: Input -> Spatial Attn (RFB+SimAM) -> Channel Attn (Linear)
      3. Output = Input + Attention_Result (Residual Add)
    """
    def __init__(self, c1, *args):
        super(RFB_CBAM_Spatial, self).__init__()
        
        # --- 參數解析 ---
        if len(args) >= 1: c2 = args[0]
        else: c2 = c1 
        if len(args) >= 2: r = args[1]
        else: r = 2 
        
        # 殘差連結要求輸入輸出通道必須一致，否則無法相加
        if c1 != c2:
            raise ValueError(f"開啟殘差連結時，輸入(c1={c1})和輸出(c2={c2})通道數必須相等。")
        
        # 1. 空間注意力組件 (RFB-SAM)
        self.branch1 = nn.Conv2d(2, 1, 1) 
        self.branch2 = nn.Conv2d(2, 1, 3, padding=1)  
        self.branch3 = nn.Conv2d(2, 1, 3, padding=2, dilation=2)
        self.reduce_conv = nn.Conv2d(3, 1, 1)
        
        # 2. 通道注意力組件 (Linear Version)
        self.channel_attention = ChannelAttention(c1, r)

    def forward(self, x):
        # 保存原始輸入作為殘差
        residual = x

        # ==================================================
        # 步驟 1: 計算空間注意力 (RFB + SimAM)
        # ==================================================
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        x_pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        s_merged = torch.cat([
            self.branch1(x_pooled),
            self.branch2(x_pooled),
            self.branch3(x_pooled)
        ], dim=1)
        s_reduced = self.reduce_conv(s_merged)

        # SimAM 能量函數計算
        N = s_reduced.shape[2] * s_reduced.shape[3] - 1
        mean = s_reduced.mean(dim=[2, 3], keepdim=True)
        var = (s_reduced - mean).pow(2).sum(dim=[2, 3], keepdim=True) / N
        e_t = 4 * (var + 1e-4) / ((s_reduced - mean).pow(2) + 2 * var + 2 * 1e-4)
        
        # 產生空間權重
        spatial_weights = (torch.sigmoid(1.0 / (e_t + 1e-4)) - 0.5) * 2
        
        # 應用空間權重 (Spatial Applied)
        x_spatial = x * spatial_weights.expand_as(x)
        
        # ==================================================
        # 步驟 2: 計算並應用通道注意力 (Channel Second)
        # ==================================================
        channel_weights = self.channel_attention(x_spatial)
        
        # 這是經過雙重注意力過濾後的特徵
        attention_out = x_spatial * channel_weights
        
        # ==================================================
        # 步驟 3: 殘差相加 (Residual Connection)
        # ==================================================
        # Output = Original + Attention_Filtered
        output = residual + attention_out
        
        return output