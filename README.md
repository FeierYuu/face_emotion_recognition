# 😊 面部情绪识别系统

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于深度学习的高效面部情绪识别系统，支持实时摄像头识别、图像识别和视频分析，专为多种平台优化，包括Apple M1/M2芯片。

## ✨ 主要特性

- 🎯 **高精度识别**：基于FER2013数据集训练的深度学习模型
- 🚀 **实时处理**：支持摄像头实时情绪识别
- 📱 **多平台支持**：优化支持Apple M1/M2芯片
- 🎨 **中文界面**：完整的中文标签和界面支持
- 🔧 **易于使用**：简单的命令行接口
- 📊 **详细分析**：提供置信度和概率分布

## 系统功能

- 🔍 **实时情绪识别**：通过摄像头进行实时人脸检测和情绪分析
- 📷 **图像/视频情绪识别**：支持从本地图像和视频文件中识别情绪
- 🏷️ **中文标签支持**：所有情绪标签和界面文本均支持中文显示
- 💻 **M1芯片优化**：针对Apple M1/M2系列芯片进行了性能优化
- 📊 **支持7种情绪类别**：
  - 😊 开心 (Happy)
  - 😐 中性 (Neutral)
  - 😢 伤心 (Sad)
  - 😠 生气 (Angry)
  - 😨 恐惧 (Fear)
  - 😲 惊讶 (Surprise)
  - 🤢 厌恶 (Disgust)

## 项目结构

```
face_emotion_recognition/
├── model.py               # 模型定义与创建
├── data_preprocessing.py  # 数据预处理模块
├── unified_emotion_recognition.py # 统一情绪识别功能，支持所有平台
├── utils.py               # 工具函数和辅助类
├── combined_train.py      # 增强版训练脚本，支持多种优化策略
├── models/                # 存储预训练模型
└── datasSource/           # 数据集存储位置
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- TensorFlow 2.x
- OpenCV 4.x
- NumPy
- 其他依赖见 `requirements.txt`

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/yourusername/face_emotion_recognition.git
   cd face_emotion_recognition
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **下载预训练模型**
   ```bash
   # 模型文件已包含在项目中
   # 如果需要重新训练，请参考训练指南
   ```

4. **运行测试**
   ```bash
   # 测试图像识别
   python unified_emotion_recognition.py --image images/90.jpg
   
   # 启动实时摄像头识别
   python unified_emotion_recognition.py --camera
   ```

## 📖 使用指南

### 🎥 实时摄像头识别

```bash
# 基础使用（默认摄像头）
python unified_emotion_recognition.py

# 使用TensorFlow Lite模型（推荐）
python unified_emotion_recognition.py --use_tflite

# 调整摄像头分辨率
python unified_emotion_recognition.py --resolution 1280 720

# 启用M1/M2芯片优化
python unified_emotion_recognition.py --use_m1_optimizations
```

### 📷 图像识别

```bash
# 识别单张图像
python unified_emotion_recognition.py --image path/to/image.jpg

# 调整置信度阈值
python unified_emotion_recognition.py --image path/to/image.jpg --confidence 0.5

# 保存识别结果
python unified_emotion_recognition.py --image path/to/image.jpg --save_result
```

### 🎬 视频识别

```bash
# 识别视频文件
python unified_emotion_recognition.py --video path/to/video.mp4

# 显示处理帧率
python unified_emotion_recognition.py --video path/to/video.mp4 --show_fps
```

### ⚙️ 高级参数

```bash
# 使用Keras模型（如果TFLite效果不佳）
python unified_emotion_recognition.py --use-keras

# 启用帧跳过策略（提高性能）
python unified_emotion_recognition.py --skip_frames 2

# 详细输出模式
python unified_emotion_recognition.py --verbose
```

### 调整识别精度

- 提高置信度阈值可减少误识别，但可能会降低检测率
- 降低置信度阈值可提高检测率，但可能增加误识别
- 默认置信度阈值为0.3

## 模型训练

### 使用增强版训练脚本

```bash
# 从头训练(推荐配置)
python combined_train.py --fer2013 datasSource/fer2013.csv --epochs 100 --batch_size 32 --learning_rate 0.0001 --augment_level high --class_weight --optimizer adam --experiment_name emotion_model_optimized

# 从已有模型继续训练
python combined_train.py --model models/your_model.h5 --fer2013 datasSource/fer2013.csv --epochs 50 --batch_size 32 --learning_rate 0.00001 --freeze_layers 2

# 简化配置训练
python combined_train.py --fer2013 datasSource/fer2013.csv --epochs 50
```

### 训练参数详解

- `--dataset`/`--fer2013`：数据集路径
- `--model`：可选，已有模型路径，用于继续训练
- `--epochs`：训练轮数，默认为50
- `--learning_rate`：学习率，默认为0.001
- `--batch_size`：批次大小，默认为64
- `--augment_level`：数据增强级别 (low/medium/high/very_high)
- `--optimizer`：优化器选择 (adam/rmsprop/sgd)
- `--freeze_layers`：冻结层数，用于微调
- `--class_weight`：启用类别权重平衡
- `--experiment_name`：实验名称，用于保存结果

### 提高训练精度的技巧

1. **调整学习率**：根据训练曲线动态调整学习率
2. **使用数据增强**：增加`augment_level`参数值，增强数据多样性
3. **平衡类别权重**：启用`--class_weight`参数解决类别不平衡问题
4. **模型微调**：先冻结低层，只训练高层，然后逐步解冻
5. **使用EarlyStopping**：防止过拟合
6. **监控验证指标**：关注验证集上的准确率和损失变化

### M1 Pro芯片优化训练

- 在M1芯片上使用TensorFlow Metal插件加速训练
- 适当调整批次大小以充分利用GPU内存
- 使用优化的数据加载方式减少CPU-GPU数据传输瓶颈

## 数据集分析

### FER2013数据集类别分布

- 😊 开心: 25.05% (约8,974张图像)
- 😐 中性: 17.27% (约6,198张图像)
- 😢 伤心: 14.27% (约5,121张图像)
- 😠 生气: 13.1% (约4,708张图像)
- 😨 恐惧: 12.44% (约4,477张图像)
- 😲 惊讶: 11.37% (约4,097张图像)
- 🤢 厌恶: 1.52% (约547张图像)

### 类别不平衡问题解决

1. **降低置信度阈值**：对于样本较少的类别（如厌恶），可适当降低其检测阈值
2. **增加训练轮数**：延长训练时间以提高模型对少数类别的学习效果
3. **启用数据增强**：为少数类别生成更多训练样本
4. **添加类别权重**：在损失函数中为少数类别分配更高的权重
5. **改进模型架构**：增加模型容量以更好地学习少数类别的特征
6. **收集更多数据**：补充少数类别的样本数量

## 模型性能评估

训练完成后，系统会自动生成以下评估结果：

- 训练和验证的准确率、损失曲线
- 混淆矩阵可视化
- 分类报告（精确率、召回率、F1分数）
- 模型保存（最佳模型和最终模型）

使用TensorBoard查看详细训练过程：
```bash
tensorboard --logdir logs/
```

## M1/M2芯片优化功能

unified_emotion_recognition.py已针对Apple Silicon芯片(M1/M2系列)进行了全面优化：

1. **TensorFlow Metal加速**：自动检测并启用GPU加速
2. **TensorFlow Lite支持**：通过`--use_tflite`参数启用轻量级模型推理
3. **帧跳过策略**：通过`--skip_frames`参数减少处理帧数，提高实时性能
4. **优化的资源管理**：智能分配内存和计算资源
5. **批量处理优化**：高效处理多个人脸场景

### M1/M2芯片专用训练参数

```bash
# M1/M2芯片上的优化训练配置
python combined_train.py --fer2013 datasSource/fer2013.csv --epochs 100 --batch_size 32 --learning_rate 0.0001 --augment_level medium --class_weight --optimizer adam --experiment_name m1_optimized_model
```

## 常见问题解决

1. **人脸检测失败**
   - 确保光线充足
   - 人脸正对摄像头
   - 尝试调整摄像头分辨率

2. **情绪识别不准确**
   - 这可能是由于参数设置或模型选择导致的。我们提供了两个工具来帮助解决这个问题：
     
     ### 配置恢复工具
     该工具可以帮助您调整情绪识别系统的设置，以恢复识别准确性：
     ```bash
     # 查看帮助信息
     ./restore_accuracy_settings.py --help
     
     # 完全恢复到原始设置
     ./restore_accuracy_settings.py --restore-original
     
     # 调整置信度阈值（0.3=高灵敏度，0.5=高精度）
     ./restore_accuracy_settings.py --threshold 0.5
     
     # 调整人脸检测参数（建议值: 1.1 5 用于高精度）
     ./restore_accuracy_settings.py --face-detection 1.1 5
     ```
     
     ### 参数测试工具
     该工具允许您在同一张图像上测试不同的参数设置，比较识别结果：
     ```bash
     # 测试默认图像
     ./test_different_settings.py
     
     # 测试特定图像
     ./test_different_settings.py --image images/90.jpg
     
     # 测试并保存结果
     ./test_different_settings.py --image images/90.jpg --save
     ```
     
     ### 提高识别准确性的建议
     1. **使用较高的置信度阈值**（如0.5）可以提高识别的准确性
     2. **使用更严格的人脸检测参数**可以减少误识别
     3. **确保光线充足**，避免背光或光线过暗的环境
     4. **保持面部表情自然**，避免夸张或不自然的表情
     5. **尝试不同的模型**，看哪个最适合您的需求
     
   ### 视频识别中人脸框时有时无
   如果您在视频识别模式下遇到人脸框时有时无或情绪识别不准确的问题，可以使用视频识别优化工具：
   
   ```bash
   # 应用所有视频识别优化（推荐）
   python enhance_video_recognition.py --all
   
   # 仅优化人脸检测稳定性
   python enhance_video_recognition.py --optimize-face-detection
   
   # 修复中性表情被误判为惊讶的问题
   python enhance_video_recognition.py --adjust-neutral-threshold
   
   # 启用人脸跟踪功能
   python enhance_video_recognition.py --enable-face-tracking
   ```
   
   优化后推荐的运行命令：
   ```bash
   # 使用平衡的置信度阈值
   python unified_emotion_recognition.py --confidence 0.4
   
   # 如果TFLite模型效果不佳，尝试使用Keras模型
   python unified_emotion_recognition.py --use-keras
   ```
   
   这些优化包括：
   - 添加备用人脸检测策略，在严格参数检测不到时自动使用宽松参数
   - 降低中性情绪阈值，减少面无表情被误判为惊讶的情况
   - 启用人脸跟踪功能，在短时间内保持人脸框的连续性
   - 支持切换回原始Keras模型（如果TFLite模型效果不理想）

3. **M1芯片上运行缓慢**
   - 使用`--use_tflite`参数启用TensorFlow Lite加速
   - 增加`--skip_frames`值减少处理帧数
   - 降低摄像头分辨率

4. **训练过程中断**
   - 检查内存是否充足，减小批次大小
   - 确保数据集路径正确
   - 如遇CUDA相关错误，请检查TensorFlow和CUDA版本兼容性
   - 在M1/M2芯片上，系统会自动处理GPU加速配置

5. **中文显示问题**
   - 确保系统已安装中文字体
   - 在macOS上，系统会自动处理字体加载

## 注意事项

- 本系统在光线良好的环境下性能最佳
- 识别精度可能受面部表情复杂度、遮挡物等因素影响
- 长时间运行可能会导致内存占用增加，建议定期重启
- 对于隐私保护，请确保在使用摄像头功能时获得相关人员的同意

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [FER2013数据集](https://www.kaggle.com/datasets/msambare/fer2013)
- [TensorFlow](https://tensorflow.org)
- [OpenCV](https://opencv.org)
- [Keras](https://keras.io)

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/yourusername/face_emotion_recognition/issues)
- 发送邮件至：your-email@example.com

---

⭐ 如果这个项目对你有帮助，请给它一个星标！