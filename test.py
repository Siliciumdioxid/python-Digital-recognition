import torch
import torch.nn as nn
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk, ImageGrab
import numpy as np
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.autograd import Variable

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),  # 输入1通道，输出16通道，卷积核大小5x5，步长1，填充2
            nn.ReLU(),
            nn.MaxPool2d(2)  # 2x2池化层
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # 输入16通道，输出32通道
            nn.ReLU(),
            nn.MaxPool2d(2)  # 2x2池化层
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输入32*7*7，输出10类（0-9）

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        output = self.out(x)
        return output

# 加载 MNIST 数据集
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./data/',  
    train=True,  
    transform=transforms.ToTensor(),  
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(
    root='./data/',
    train=False
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=50,
    shuffle=True
)

# 训练模型
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001

cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


# 训练循环
# 首次使用需要先训练一个模型存放在本地，训练好后可将以下代码注释
#---------------
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch数据
        output = cnn(b_x)  # 输入数据到CNN中计算output
        loss = loss_func(output, b_y)  # 计算损失
        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新梯度

        if step % 50 == 0:
            test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
            test_y = test_data.targets[:2000]

            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print(f'Epoch: {epoch} | train loss: {loss.item():.4f} | test accuracy: {accuracy:.2f}')

# 保存训练好的模型
torch.save(cnn.state_dict(), 'cnn2.pkl')  # 保存模型权重
#---------------


# 加载训练好的模型
cnn.load_state_dict(torch.load('cnn2.pkl'))
cnn.eval()  # 设为评估模式

# 手写数字识别部分
def predict_digit(image):
    # 转换为灰度图像并调整为28x28大小
    image = image.convert('L').resize((28, 28))
    image = np.array(image) / 255.0  # 归一化到0-1
    image = torch.Tensor(image).unsqueeze(0).unsqueeze(0)  # 转换为 (1,1,28,28)
    
    with torch.no_grad():  # 禁用梯度计算
        output = cnn(image)
        _, predicted = torch.max(output, 1)  # 获取最大值对应的标签
    return predicted.item()

# 创建Tkinter窗口
window = tk.Tk()
window.title("手写数字识别")

# 创建画布，用于绘制数字
canvas = Canvas(window, width=280, height=280, bg='black')  # 黑色背景
canvas.grid(row=0, column=0)

# 用于存储手写数字的绘制路径
last_x, last_y = None, None

# 清空画布并清空识别结果
def clear_canvas():
    canvas.delete('all')  # 清空画布
    result_label.config(text="识别结果: ")  # 清空识别结果标签中的内容

# 处理绘制动作
def paint(event):
    global last_x, last_y
    x, y = event.x, event.y
    if last_x and last_y:
        canvas.create_line(last_x, last_y, x, y, width=8, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE)  # 白色笔画
    last_x, last_y = x, y

# 重置上一点坐标
def reset(event):
    global last_x, last_y
    last_x, last_y = None, None

# 识别并显示结果
def recognize_digit():
    # 从画布上截取图像并转换为图片
    x1 = canvas.winfo_rootx()
    y1 = canvas.winfo_rooty()
    x2 = x1 + canvas.winfo_width()
    y2 = y1 + canvas.winfo_height()
    
    # 使用Pillow的ImageGrab来截取画布内容
    image = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    label = predict_digit(image)  # 使用CNN模型进行识别
    result_label.config(text=f"识别结果: {label}")  # 显示识别结果

# 创建按钮和标签
clear_button = tk.Button(window, text="清空画布", command=clear_canvas)
clear_button.grid(row=1, column=0)

recognize_button = tk.Button(window, text="识别数字", command=recognize_digit)
recognize_button.grid(row=2, column=0)

result_label = tk.Label(window, text="识别结果: ")
result_label.grid(row=3, column=0)

# 绑定鼠标事件
canvas.bind("<B1-Motion>", paint)  # 按住左键时绘制
canvas.bind("<ButtonRelease-1>", reset)  # 释放鼠标左键时重置

# 启动Tkinter主循环
window.mainloop()
