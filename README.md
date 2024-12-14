# 手写数字识别项目
## 项目简介
本项目实现了一个基于卷积神经网络（CNN）的手写数字识别系统。用户可以在一个简单的图形用户界面（GUI）中手写数字，系统会自动识别并显示识别结果。该项目使用了PyTorch框架进行模型训练和推理，并使用Tkinter库构建GUI。
## 目录结构
```
project/
├── cnn2.pkl  # 训练好的模型权重
├── main.py   # 主程序文件
└── README.md # 项目说明文件
```
## 环境依赖
- Python 3.7+
- PyTorch 1.7+
- torchvision
- tkinter
- PIL (Pillow)
- numpy
## 安装依赖
在项目根目录下运行以下命令来安装所需的依赖：
```sh
pip install torch torchvision pillow numpy
```
## 数据集
本项目使用MNIST数据集进行训练和测试。如果数据集不存在，程序会自动下载。
## 运行项目
1. **训练模型**（首次使用时需要训练模型）：
   - 打开 `main.py` 文件，取消注释训练部分的代码。
   - 运行 `main.py` 文件：
     ```sh
     python main.py
     ```
   - 训练完成后，模型权重将保存在 `cnn2.pkl` 文件中。
2. **加载模型并运行GUI**：
   - 确保 `cnn2.pkl` 文件存在。
   - 运行 `main.py` 文件：
     ```sh
     python main.py
     ```
   - 打开的窗口中，用户可以在画布上手写数字，点击“识别数字”按钮后，系统会显示识别结果。点击“清空画布”按钮可以清空画布并重置识别结果。
## 代码结构
### 主程序文件 `main.py`
1. **定义CNN模型**：
   - `CNN` 类定义了一个包含两个卷积层和一个全连接层的卷积神经网络。
   - `forward` 方法定义了前向传播过程。
2. **加载MNIST数据集**：
   - 使用 `torchvision.datasets.MNIST` 加载MNIST数据集。
   - `train_data` 和 `test_data` 分别用于训练和测试。
   - `train_loader` 用于批量加载训练数据。
3. **训练模型**：
   - 使用 `torch.optim.Adam` 优化器和 `nn.CrossEntropyLoss` 损失函数进行训练。
   - 训练过程中，每50个批次输出一次训练损失和测试准确率。
   - 训练完成后，保存模型权重到 `cnn2.pkl` 文件中。
4. **加载训练好的模型**：
   - 使用 `torch.load` 加载模型权重并设为评估模式。
5. **手写数字识别部分**：
   - `predict_digit` 函数将画布上的图像转换为28x28的灰度图像，并使用CNN模型进行识别。
   - `clear_canvas` 函数清空画布和识别结果标签。
   - `paint` 和 `reset` 函数处理鼠标绘制动作。
   - `recognize_digit` 函数从画布上截取图像并调用 `predict_digit` 进行识别。
6. **创建Tkinter窗口**：
   - 使用 `tkinter` 创建一个包含画布、按钮和标签的GUI。
   - 绑定鼠标事件以实现手写绘制和识别功能。
## 注意事项
- 确保在首次使用时训练模型并保存权重。
- 如果遇到任何问题，请检查依赖是否安装正确，并确保数据集下载成功。
