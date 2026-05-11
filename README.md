# Optimization of Small Object Detection in UAV Aerial Imagery

## 🚀 專案簡介 (Overview)
本研究針對無人機（UAV）高空視角下的「小目標檢測」任務，開發了一套基於 **YOLOv11** 的改良架構。透過整合 **BiFPN 加權特徵融合機制** 與 **SPD-Conv 無損下採樣模組**，本模型能更有效地保留微小目標的特徵資訊，同時維持輕量化優勢。

*   **mAP@50**: 達到 **50.40%**，相較於原版 YOLOv11-s 提升了 **10.09%**。
*   **模型尺寸**: 參數量僅微幅增加至 **12.02M**，證明了設計的高效性。

## 核心改進 (Key Innovations)

*   **BiFPN (Weighted Feature Fusion)**: 採用雙向加權特徵金字塔結構，能更有效地融合深層語義與淺層細節特徵，且特徵融合過程更具效率。
*   **SPD-Conv (Lossless Downsampling)**: 藉由 **Space-to-Depth** 機制取代會造成資訊流失的步長卷積，使模型在連續下採樣後仍能保留小物件的細微邊界與紋理資訊，解決小目標「特徵消失」的痛點。

## 資料夾結構 (Repository Structure)：
```bash
├── api/                #FastAPI 服務
│   ├── __init__.py
│   ├── app.py
│   ├── detector.py
│   └── main.py
├── data/               # 資料集路徑
├── models/             # 模型架構定義 (YAML)
│   ├── yolo11_BiFPN_s.yaml
│   ├── yolo11_BiFPN_SPDConv.yaml
│   └── yolo11_SPD.yaml
├── ultralytics/        # 核心運算引擎包
├── weights/            # 訓練權重存儲
│   ├── BiFPN/
│   ├── Proposed/
│   ├── SPDConv/
│   └── YOLOv11/
├── .gitignore          # 版本控制忽略清單
├── README.md           # 專案說明文件
└── requirements.txt    # 環境依賴清單
```

## 實驗結果 (Performance Comparison)

本專案在 **VisDrone-DET2019** 驗證集上進行測試，結果顯示結合架構改良與推理策略後，模型效能獲得顯著提升。

| 模型架構組合 | mAP@50 (%) | 提升幅度 (Improvement) | 參數量 (Params) |
| :--- | :---: | :---: | :---: |
| YOLOv11s (Baseline) | 40.31 | - | 9.43M |
| YOLOv11s + BiFPN | 47.07 | +6.76 | 8.06M |
| YOLOv11s + SPD-Conv | 42.12 | +1.81 | 10.62M |
| **UAV-YOLOv11 (Proposed)** | **50.40** | **+10.09** | **12.02M** |

**Note**：
1. **架構改良**：引入 BiFPN 與 SPD-Conv 後，純模型效能從 40.31% 提升至 50.40%。
2. **輕量化**：在大幅提升精度的同時，參數量僅微幅增加至 12.02M，極具邊緣端部署潛力。