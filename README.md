# Project-SMS

為了解決簡訊詐騙而自製的小型簡訊分類模型

## 近期變動
### 3.0.0（2024 年 1 月 4 日）
- 修正了main.py沒有成功調用NVIDIA CUDA 訓練的錯誤
- 新增了batch(批次訓練)的功能
- 調整了模型的架構 : 輸入層 → 第一個隱藏層 (fc1) → 第二個隱藏層 (fc2) → 輸出層
- 刪除了原始碼中訓練生成檔

### 2.2.0（2024 年 1 月 4 日）
- 修正了main.py沒有成功調用NVIDIA CUDA 訓練的錯誤

### 2.1.2（2024 年 1 月 1 日）
- 更新 SMS_data.json 的資料筆數

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

### 檔案說明
- 必要檔案
  main.py:訓練程式
  test.py:測試程式
  SMS_data.json:訓練資料庫
  
- 附加檔案(透過main.py生成)
  config.json:模型配置文件
  labels.txt:標籤文件
  SMS_model.bin:模型
  tokenizer.json:詞彙表

### 安裝
```shell
git clone https://github.com/AmanoShizukikun/Project-SMS.git
cd Project-SMS
```

- 修改訓練資料庫
```shell
.\SMS_data.json
```

- 開始訓練
```shell
python main.py
```

- 開始測試
```shell
python test.py
```
