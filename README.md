# Project-SMS

為了解決簡訊詐騙而自製的小型簡訊分類模型

## 近期變動
### 2.1.1（2023 年 12 月 31 日）
- 修正了main.py的錯誤，現在只要在SMS_data.json的output中新增新的標籤，程式就會自動添加新的標籤

### 2.1.1（2023 年 12 月 30 日）
- 將SMS_data.json的output改為英文修正了labels.txt出現亂碼的情形
- 修改訓練過程的步驟顯示方式，能更直觀的看到訓練的過程了
- 修改main.py的標籤映射(label_mapping)的方式

### 2.0.0（2023 年 12 月 24 日）
- 這個版本從上一版只能判斷單一類型簡訊改為可以簡易分類簡訊類型

## 快速開始
 **粗體** 的是強制要求的。
 
### 硬體要求
1. 作業系統：Windows
1. **CPU** / Nvidia GPU

### 環境設置
- **Python 3**
- 下載: https://www.python.org/downloads/windows/
- **PyTorch**
- 下載: https://pytorch.org/
- NVIDIA GPU驅動程式
- 下載: https://www.nvidia.com/zh-tw/geforce/drivers/
- NVIDIA CUDA Toolkit
- 下載: https://developer.nvidia.com/cuda-toolkit
- NVIDIA cuDNN
- 下載: https://developer.nvidia.com/cudnn
