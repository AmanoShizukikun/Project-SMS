import os
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

# 設備選擇如果有NVIDIA顯卡切換為CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 環境檢測
print("<環境檢測>")
print(torch.__version__)
print(device)

# 取得目前執行程式的資料夾路徑
current_directory = Path(__file__).resolve().parent

# 定義神經網路模型
class SMSClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SMSClassifier, self).__init__()

        # 輸入層到第一個隱藏層
        self.fc1 = nn.Linear(input_size, int(hidden_size * 0.66)) 
        self.relu1 = nn.ReLU()

        # 第一個隱藏層到第二個隱藏層
        self.fc2 = nn.Linear(int(hidden_size * 0.66), int(hidden_size * 0.66)) 
        self.relu2 = nn.ReLU()

        # 最後一個隱藏層到輸出層
        self.fc3 = nn.Linear(int(hidden_size * 0.66), output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)  # 輸入層到第一個隱藏層
        x = self.relu1(x)
        x = self.fc2(x)  # 第一個隱藏層到第二個隱藏層
        x = self.relu2(x)
        x = self.fc3(x)  # 最後一個隱藏層到輸出層
        x = self.softmax(x)
        return x

# 將文本轉換為向量
def text_to_vector(text):
    vector = [0] * len(vocab)
    for word in text:
        if word in vocab:
            vector[vocab.index(word)] = 1
    return vector

# 加載模型和 vocab
def load_model(model_path, vocab_path, config_path, label_path, device):
    # 讀取 vocab
    with open(vocab_path, 'r') as json_file:
        vocab = json.load(json_file)
        
    # 讀取模型配置
    with open(config_path, 'r') as json_file:
        model_config = json.load(json_file)
        
    # 讀取標籤映射
    with open(label_path, 'r') as labels_file:
        label_mapping = {}
        for line in labels_file:
            label, index = line.strip().split(': ')
            label_mapping[label] = int(index)

    # 初始化模型並將其移到 CUDA 上
    model = SMSClassifier(model_config['input_size'], model_config['hidden_size'], model_config['output_size'])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    return model, vocab, label_mapping

# 設定檔案路徑
VOCAB_PATH = current_directory /"models"/ "tokenizer.json"
MODEL_PATH = current_directory /"models"/ "SMS_model.bin"
CONFIG_PATH = current_directory /"models"/ "config.json"
LABEL_PATH = current_directory /"models"/ "labels.txt"


# 加载模型、vocab和標籤映射
model, vocab, label_mapping = load_model(MODEL_PATH, VOCAB_PATH, CONFIG_PATH, LABEL_PATH, device)

# 測試模型
def predict_SMS(text):
    input_vector = text_to_vector(text)  # 取得文字向量
    input_vector = torch.tensor(input_vector, dtype=torch.float).unsqueeze(0).to(device)  # 調整張量形狀並移到裝置上
    output = model(input_vector)
    predicted_class = torch.argmax(output).item()
    predicted_probs = output.squeeze().tolist()  # 預測機率列表
    predicted_label = [label for label, index in label_mapping.items() if index == predicted_class][0]
    print(f"'{text}'  預測概率: {predicted_probs}")
    print(f"預測結果: {predicted_label}")
    return predicted_label, predicted_probs, predicted_class

# 用戶輸入
print("<測試開始>")
while True:
    try:
        user_input = input("請輸入簡訊內容（按下 Enter 結束程式）：")
        if user_input == "":
            print("程式結束。")
            break  # 如果輸入為空，則停止迴圈並結束程式
        predicted_label, predicted_probs, predicted_class = predict_SMS(user_input)
    except KeyboardInterrupt:
        print("程式已手動停止。")
        break  # 如果按下 Ctrl+C，則停止迴圈並結束程式