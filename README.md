# 手写数字识别系统

## 项目简介

该项目实现了一个**手写数字识别系统**，用户可以通过**Tkinter图形界面**手动绘制数字，系统将使用训练好的**卷积神经网络（CNN）**模型对用户绘制的数字进行识别。整个系统包含了**模型训练**、**图形界面设计**以及**数字识别**功能。

## 功能特点

- **训练CNN模型**：使用MNIST手写数字数据集训练卷积神经网络进行数字分类。
- **图形界面**：使用Tkinter创建画布，用户可以手动绘制数字，背景为黑色，数字为白色。
- **数字识别**：用户绘制完数字后，可以通过点击按钮识别数字，识别结果会显示在界面上。
- **模型保存与加载**：训练完成后，保存训练好的模型，并支持加载已训练的模型进行数字识别。

## 环境要求

- Python 3.x
- PyTorch
- Tkinter
- PIL（Python Imaging Library）

### 安装依赖

1. 安装PyTorch：  
   请根据您的操作系统和硬件环境（如是否有GPU支持）安装合适的PyTorch版本。  
   官方网站：[PyTorch官网](https://pytorch.org/get-started/locally/)

2. 安装其他依赖：
   ```bash
   pip install torchvision matplotlib Pillow
   ```

3. 如果您遇到`ImageGrab`模块的问题，请安装`pillow`库来支持截图功能：
   ```bash
   pip install pillow
   ```

## 使用方法

1. **训练模型**：  
   运行`train_model.py`文件进行模型训练。训练过程中会输出损失值和测试准确率，并在训练结束后保存模型到文件`cnn2.pkl`。

2. **运行数字识别界面**：  
   运行`recognize_digit_gui.py`，启动一个图形界面，用户可以在画布上绘制数字，并点击“识别数字”按钮查看识别结果。

3. **清空画布**：  
   用户可以点击“清空画布”按钮来清除画布上的内容，准备绘制新的数字。

4. **识别结果**：  
   用户绘制完数字后，点击“识别数字”按钮，系统会识别该数字并显示在界面上。

## 文件结构

```
project/
│
├── train_model.py            # 训练CNN模型的脚本
├── recognize_digit_gui.py     # 手写数字识别图形界面
├── cnn2.pkl                  # 训练好的模型（由train_model.py生成）
├── README.md                 # 本文档
└── requirements.txt          # 项目依赖列表
```

### 训练模型脚本（`train_model.py`）：

该脚本会训练卷积神经网络，并将训练好的模型保存到`cnn2.pkl`。

```python
import torch
import torchvision
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# 数据加载与处理
train_data = torchvision.datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = Data.DataLoader(dataset=train_data, batch_size=50, shuffle=True)

# 网络定义
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 训练设置
cnn = CNN()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

# 训练过程
for epoch in range(5):  # 训练5个epoch
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 保存训练好的模型
torch.save(cnn.state_dict(), 'cnn2.pkl')
```

### 手写数字识别图形界面（`recognize_digit_gui.py`）：

该脚本提供了一个用户界面，允许用户绘制数字并识别。

```python
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageGrab
import torch
import torchvision.transforms as transforms
from torch import nn
import numpy as np

# 加载训练好的模型
cnn = CNN()
cnn.load_state_dict(torch.load('cnn2.pkl'))
cnn.eval()

# 图像处理和识别
def predict_digit(image):
    image = image.convert('L').resize((28, 28))  # 转为灰度图并调整大小
    image = np.array(image) / 255.0  # 归一化
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()  # 转换为tensor
    output = cnn(image)  # 使用CNN进行预测
    _, predicted = torch.max(output, 1)
    return predicted.item()

# GUI界面实现
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别")
        self.canvas = tk.Canvas(self.root, width=200, height=200, bg="black")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("RGB", (200, 200), color="black")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label = tk.Label(self.root, text="识别结果：", font=("Arial", 14))
        self.result_label.pack()
        self.clear_button = tk.Button(self.root, text="清空画布", command=self.clear_canvas)
        self.clear_button.pack()
        self.recognize_button = tk.Button(self.root, text="识别数字", command=self.recognize_digit)
        self.recognize_button.pack()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", width=5)
        self.draw.line([x1, y1, x2, y2], fill="white", width=5)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (200, 200), color="black")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="识别结果：")

    def recognize_digit(self):
        img = self.image.convert("L")  # 转为灰度图
        label = predict_digit(img)  # 使用CNN模型进行识别
        self.result_label.config(text=f"识别结果：{label}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
```

## 常见问题

1. **安装问题**：
   - 确保你已经安装了所有的依赖库。如果遇到安装问题，请检查Python版本或更新依赖库。
   
2. **图像保存错误**：
   - 如果在保存图像时遇到问题，确保你的环境支持PIL库以及相关格式的图像保存。

3. **模型加载问题**：
   - 确保`cnn2.pkl`文件存在，并且文件路径正确。如果你没有训练模型，可以运行`train_model.py`脚本来训练并保存模型。

## 许可证

该项目使用MIT许可证，详细内容请见`LICENSE`文件。
