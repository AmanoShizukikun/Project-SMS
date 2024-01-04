import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from tqdm import tqdm
import time

# 檢測設備是否有 NVIDIA 顯卡，選擇使用 CUDA 還是 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 環境檢測
print("<環境檢測>")
print(f"PyTorch版本 : {torch.__version__}")
print(f"訓練設備 : {device}")

# 生成詞彙表
print("<詞彙表生成>")
# 獲取當前運行程序的文件夾路徑
current_directory = os.path.dirname(os.path.abspath(__file__))
# 載入訓練數據
json_model_path = os.path.join(current_directory, 'SMS_data.json')
with open(json_model_path, 'r', encoding='utf-8') as json_file:
    model_data = json.load(json_file)
# 提取訓練數據
instructions = [item["instruction"] for item in model_data]
outputs = [item["output"] for item in model_data]

# 提取詞彙表文字
vocab_text = []
for value in model_data:
    if 'weight' in value:
        vocab_text.extend(value)
        
# 自動生成標籤編號
print("<類別標籤生成>")
label_mapping = {}
label_count = 0
for label in outputs:
    if label not in label_mapping:
        label_mapping[label] = label_count
        label_count += 1

# 將標籤儲存為 labels.txt
labels_path = os.path.join(current_directory, 'labels.txt')
with open(labels_path, 'w') as labels_file:
    for label, index in label_mapping.items():
        labels_file.write(f"{label}: {index}\n")
print(f"標籤已儲存於 {labels_path}")
print(f"類別標籤生成完成 Lables 儲存為 .txt 於 {labels_path}")

# 創建詞彙表 vocab
vocab = list(set(''.join([text for text in instructions])))
# 將 vocab 保存為 tokenizer.json
tokenizer_path = os.path.join(current_directory, 'tokenizer.json')
with open(tokenizer_path, 'w') as json_file:
    json.dump(vocab, json_file, indent=2)
print(f"詞彙表生成完成 Tokenizer 儲存為 .json 於 {tokenizer_path}")

# 將文本轉換為向量
def text_to_vector(text):
    vector = [0] * len(vocab)
    for word in text:
        if word in vocab:
            vector[vocab.index(word)] = 1
    return vector

# 將數據轉換為模型可用的格式
train_data = [(text_to_vector(text), label_mapping[label]) for text, label in zip(instructions, outputs)]

print("<訓練開始>")

# 定義神經網路模型
class SMSClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SMSClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# 設置模型參數
input_size = len(vocab)
hidden_size = 4096
output_size = len(label_mapping)

# 初始化模型、損失函數和優化器
model = SMSClassifier(input_size, hidden_size, output_size)
model = model.to(device)  # 將模型移動到 GPU
criterion = nn.CrossEntropyLoss()  # 使用 CrossEntropyLoss 作為損失函數
criterion = criterion.to(device)  # 將損失函數移動到 GPU
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# 記錄開始訓練時間
training_start_time = time.time()

# 訓練模型
epochs = 500
for epoch in range(epochs):
    total_loss = 0
    start_time = time.time()
    
    for text_vector, label in train_data:  # 更新迭代變數為兩個值的元組
        optimizer.zero_grad()
        inputs = torch.tensor(text_vector, dtype=torch.float).to(device)
        label = torch.tensor(label, dtype=torch.long).to(device)
        output = model(inputs)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    # 輸出每個 epoch 的平均損失
    average_loss = total_loss / len(train_data)
    
    # 計算運行時間
    elapsed_time = time.time() - start_time
    # 計算預計完成時間
    eta = (epochs - epoch - 1) * elapsed_time
    # 計算總訓練時間
    total_training_time = time.time() - training_start_time
    
     # 動態計算空格填充數量
    progress = epoch + 1
    percentage = progress / epochs * 100
    fill_length = int(50 * progress / epochs)
    space_length = 50 - fill_length
    print(f"Processing: {percentage:3.0f}%|{'█' * fill_length}{' ' * space_length}| {progress}/{epochs} [{total_training_time:.2f}<{eta:.2f}, {1 / elapsed_time:.2f}it/s, Loss: {average_loss:.4f}] ")
    
# 訓練結束 
print("訓練完成")

print("<生成模型配置文件>")
# 定義模型配置
model_config = {
    "_name_or_path": "Project SMS",
    "model_type": "Project SMS",
    "architectures": ["Project SMS Model"],
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "learning_rate": optimizer.param_groups[0]['lr'],
}

# 儲存模型配置為 config.json
config_path = os.path.join(current_directory, 'config.json')
with open(config_path, 'w') as json_file:
    json.dump(model_config, json_file, indent=4)
print(f"模型配置文件生成完成 模型配置文件儲存於 {config_path}")

print("<保存模型>")
model_path = os.path.join(current_directory, 'SMS_model.bin')
torch.save(model.state_dict(), model_path)
print(f"保存模型完成 模型保存於 {model_path}")

print("<載入測試模式>")
# 在載入模型權重之前定義 model_path
model_path = os.path.join(current_directory, 'SMS_model.bin')
# 載入模型並將其移到 CUDA 上
model = SMSClassifier(input_size, hidden_size, output_size)
model = model.to(device)
# 加載模型權重
model.load_state_dict(torch.load(model_path))

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

# 測試預設範例
test_sentences = [
    "OPEN POINT會員您好，驗證碼為47385，如非本人操作，建議您立即更改密碼。提醒您！勿將密碼、驗證碼交付他人以防詐騙。", 
    "友達實習說明會，請有意願參加同學於113年3月4日前報名https://reurl.cc/V4dQvQ 勤益分機2661"
    ]

for sentence in test_sentences:
    print("<測試開始>")
    predicted_label, predicted_probs, predicted_class = predict_SMS(sentence)
print("<測試完成>")
