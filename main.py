# 导入PyTorch相关模块，用于构建和训练神经网络
import torch#PyTorch 是一个开源的机器学习库，主要用于进行计算机视觉（CV）、自然语言处理（NLP）、语音识别等领域的研究和开发。
import torch.nn as nn#nn是Neural Network的简称，帮助程序员方便执行如下的与神经网络相关的行为：（1）创建神经网络（2）训练神经网络（3）保存神经网络4）恢复神经网络
import tkinter as tk#Python GUI编程(Tkinter)
from tkinter import Canvas
from PIL import Image, ImageTk, ImageGrab#要点：PIL库是一个具有强大图像处理能力的第三方库，不仅包含了丰富的像素、色彩操作功能，还可以用于图像归档和批量处理。
import numpy as np#NumPy，一言以蔽之，是Python中基于数组对象的科学计算库。它是Python语言的一个扩展程序库，支持大量的维度数组与矩阵运算，以及大量的数学函数库
import torchvision#torchvision独立于pytorch，专门用来处理图像，通常用于计算机视觉领域。
import torch.utils.data as Data#torch.utils.data.Dataset是代表自定义数据集方法的类，用户可以通过继承该类来自定义自己的数据集类，在继承时要求用户重载__len__()和__getitem__()这两个魔法方法。
import torchvision.transforms as transforms  # 导入PyTorch的图像变换工具
from torch.autograd import Variable  # 导入Variable类，用于自动求导

# 定义CNN模型（卷积神经网络（Convolutional Neural Networks，简称CNN），https://blog.csdn.net/AI_dataloads/article/details/133250229）



# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 调用父类的构造函数
        self.conv1 = nn.Sequential(  # 定义第一个卷积层
            nn.Conv2d(1, 16, 5, 1, 2),  # 输入1通道，输出16通道，卷积核大小5x5，步长1，填充2
            nn.ReLU(),  # 使用ReLU激活函数
            nn.MaxPool2d(2)  # 2x2池化层，用于降低特征图尺寸
        )
        self.conv2 = nn.Sequential(  # 定义第二个卷积层
            nn.Conv2d(16, 32, 5, 1, 2),  # 输入16通道，输出32通道
            nn.ReLU(),  # 使用ReLU激活函数
            nn.MaxPool2d(2)  # 2x2池化层
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # 定义全连接层，输入32*7*7，输出10类（0-9）

    def forward(self, x):
        x = self.conv1(x)  # 将输入数据传递给第一个卷积层
        x = self.conv2(x)  # 将数据传递给第二个卷积层
        x = x.view(x.size(0), -1)  # 将多维特征图展平为二维张量，以便输入到全连接层
        output = self.out(x)  # 将展平后的特征传递给全连接层，得到最终输出
        return output

# 加载 MNIST 数据集
DOWNLOAD_MNIST = True  # 设置是否下载MNIST数据集的标志

train_data = torchvision.datasets.MNIST(  # 加载MNIST训练数据集
    root='./data/',  # 数据集存储路径
    train=True,  # 指定加载训练集
    transform=transforms.ToTensor(),  # 将图像转换为Tensor
    download=DOWNLOAD_MNIST,  # 根据标志决定是否下载数据集
)

test_data = torchvision.datasets.MNIST(  # 加载MNIST测试数据集
    root='./data/',  # 数据集存储路径
    train=False  # 指定加载测试集
)

train_loader = Data.DataLoader(  # 创建数据加载器，用于批量加载训练数据
    dataset=train_data,  # 指定数据集
    batch_size=50,  # 设置每批数据的大小
    shuffle=True  # 是否随机打乱数据
)

# 训练模型
EPOCH = 1  # 设置训练轮数
BATCH_SIZE = 50  # 设置每批数据的大小
LR = 0.001  # 设置学习率

cnn = CNN()  # 创建CNN模型实例
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # 使用Adam优化器
loss_func = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

# 训练循环
# 首次使用需要先训练一个模型存放在本地，训练好后可将以下代码注释
#---------------
for epoch in range(EPOCH):  # 遍历训练轮数
    for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch数据
        output = cnn(b_x)  # 输入数据到CNN中计算output
        loss = loss_func(output, b_y)  # 计算损失
        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新梯度

        if step % 50 == 0:  # 每50步输出一次训练信息
            test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255  # 获取测试数据并归一化
            test_y = test_data.targets[:2000]  # 获取测试标签

            test_output = cnn(test_x)  # 使用模型对测试数据进行预测
            pred_y = torch.max(test_output, 1)[1].data.numpy()  # 获取预测结果
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))  # 计算准确率
            print(f'Epoch: {epoch} | train loss: {loss.item():.4f} | test accuracy: {accuracy:.2f}')

# 保存训练好的模型
torch.save(cnn.state_dict(), 'cnn2.pkl')  # 保存模型权重
#---------------


# 加载训练好的模型
cnn.load_state_dict(torch.load('cnn2.pkl'))  # 加载模型权重
cnn.eval()  # 设为评估模式

# 手写数字识别部分
def predict_digit(image):
    # 转换为灰度图像并调整为28x28大小
    image = image.convert('L').resize((28, 28))
    image = np.array(image) / 255.0  # 归一化到0-1
    image = torch.Tensor(image).unsqueeze(0).unsqueeze(0)  # 转换为 (1,1,28,28)
    
    with torch.no_grad():  # 禁用梯度计算
        output = cnn(image)  # 使用模型进行预测
        _, predicted = torch.max(output, 1)  # 获取最大值对应的标签
    return predicted.item()

# 创建Tkinter窗口
window = tk.Tk()  # 创建Tkinter窗口实例
window.title("手写数字识别")  # 设置窗口标题

# 创建画布，用于绘制数字
canvas = Canvas(window, width=280, height=280, bg='black')  # 创建画布，设置大小和背景颜色
canvas.grid(row=0, column=0)  # 将画布置于窗口的指定位置

# 用于存储手写数字的绘制路径
last_x, last_y = None, None

# 清空画布并清空识别结果
def clear_canvas():
    canvas.delete('all')  # 清空画布上的所有内容
    result_label.config(text="识别结果: ")  # 清空识别结果标签中的内容

# 处理绘制动作
def paint(event):
    global last_x, last_y
    x, y = event.x, event.y  # 获取当前鼠标位置
    if last_x and last_y:
        canvas.create_line(last_x, last_y, x, y, width=8, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE)  # 绘制线条
    last_x, last_y = x, y  # 更新上一个绘制点的坐标

# 重置上一点坐标
def reset(event):
    global last_x, last_y
    last_x, last_y = None, None  # 重置坐标

# 识别并显示结果
def recognize_digit():
    # 从画布上截取图像并转换为图片
    x1 = canvas.winfo_rootx()  # 获取画布在屏幕上的x坐标
    y1 = canvas.winfo_rooty()  # 获取画布在屏幕上的y坐标
    x2 = x1 + canvas.winfo_width()  # 获取画布右下角的x坐标
    y2 = y1 + canvas.winfo_height()  # 获取画布右下角的y坐标
    
    # 使用Pillow的ImageGrab来截取画布内容
    image = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    label = predict_digit(image)  # 使用CNN模型进行识别
    result_label.config(text=f"识别结果: {label}")  # 显示识别结果

# 创建按钮和标签
clear_button = tk.Button(window, text="清空画布", command=clear_canvas)  # 创建清空画布按钮
clear_button.grid(row=1, column=0)  # 将按钮置于窗口的指定位置

recognize_button = tk.Button(window,
