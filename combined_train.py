import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import EmotionModel
from data_preprocessing import DataPreprocessor
from utils import ModelUtils

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='面部情绪识别模型训练工具')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, help='数据集目录路径')
    parser.add_argument('--fer2013', type=str, help='FER2013 CSV文件路径')
    
    # 模型参数
    parser.add_argument('--model', type=str, help='已有模型路径，用于继续训练')
    parser.add_argument('--input_shape', type=int, nargs=3, default=(48, 48, 1), help='输入形状 (高度 宽度 通道)')
    parser.add_argument('--num_classes', type=int, default=7, help='情绪类别数量')
    parser.add_argument('--model_type', type=str, choices=['basic', 'advanced'], default='advanced', help='模型类型')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'rmsprop', 'sgd'], default='adam', help='优化器')
    
    # 增强和优化参数
    parser.add_argument('--augment_level', type=str, choices=['none', 'low', 'medium', 'high', 'very_high'], 
                        default='none', help='数据增强级别')
    parser.add_argument('--class_weight', action='store_true', help='启用类别权重平衡')
    parser.add_argument('--freeze_layers', type=int, default=0, help='冻结层数，0表示不冻结')
    
    # 输出和日志参数
    parser.add_argument('--experiment_name', type=str, default=None, help='实验名称')
    parser.add_argument('--log_dir', type=str, default='logs', help='TensorBoard日志目录')
    parser.add_argument('--save_dir', type=str, default='models', help='模型保存目录')
    
    return parser.parse_args()

def configure_tensorflow():
    """配置TensorFlow以提高性能"""
    # 启用内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU内存增长已启用")
        except RuntimeError as e:
            print(f"启用GPU内存增长失败: {e}")
    
    # 检查是否在M1芯片上运行
    if hasattr(tf.config, 'experimental'):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            device_details = tf.config.experimental.get_device_details(gpus[0])
            if 'METAL' in device_details.get('device_name', ''):
                print("检测到Apple Silicon芯片，启用Metal优化")

def get_augmentation_config(level):
    """根据增强级别返回数据增强配置"""
    configs = {
        'none': None,
        'low': {
            'rotation_range': 10,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'horizontal_flip': True
        },
        'medium': {
            'rotation_range': 15,
            'width_shift_range': 0.15,
            'height_shift_range': 0.15,
            'zoom_range': 0.1,
            'horizontal_flip': True
        },
        'high': {
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'zoom_range': 0.15,
            'shear_range': 0.1,
            'horizontal_flip': True,
            'fill_mode': 'nearest'
        },
        'very_high': {
            'rotation_range': 25,
            'width_shift_range': 0.25,
            'height_shift_range': 0.25,
            'zoom_range': 0.2,
            'shear_range': 0.15,
            'horizontal_flip': True,
            'vertical_flip': False,
            'brightness_range': [0.8, 1.2],
            'fill_mode': 'nearest'
        }
    }
    return configs.get(level, None)

def get_optimizer(name, learning_rate):
    """根据名称返回优化器"""
    optimizers = {
        'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    }
    return optimizers.get(name, tf.keras.optimizers.Adam(learning_rate=learning_rate))

def calculate_class_weights(y_train):
    """计算类别权重以解决类别不平衡问题"""
    # 将one-hot编码转换为标签
    y_labels = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train
    
    # 计算每个类别的样本数
    class_counts = np.bincount(y_labels)
    
    # 计算类别权重
    total_samples = len(y_labels)
    class_weights = {i: total_samples / (len(class_counts) * count) if count > 0 else 1.0 
                    for i, count in enumerate(class_counts)}
    
    # 打印类别分布和权重
    emotion_labels = ['生气', '厌恶', '恐惧', '开心', '中性', '伤心', '惊讶']
    print("\n数据集类别分布:")
    for i, count in enumerate(class_counts):
        percentage = (count / total_samples) * 100
        emotion_name = emotion_labels[i] if i < len(emotion_labels) else f'未知({i})'
        print(f"{emotion_name}: {count} 样本 ({percentage:.2f}%) - 权重: {class_weights[i]:.4f}")
    
    return class_weights

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 配置TensorFlow
    configure_tensorflow()
    
    # 创建数据预处理器
    preprocessor = DataPreprocessor()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置实验名称
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"exp_{timestamp}"
    
    # 加载数据集
    print("\n开始加载数据集...")
    try:
        if args.fer2013:
            # 从FER2013 CSV加载数据
            train_data, test_data = preprocessor.load_fer2013_dataset(args.fer2013)
            # FER2013没有验证集，从训练集中划分一部分作为验证集
            x_train, y_train = train_data
            x_test, y_test = test_data
            
            # 划分验证集
            val_split = 0.1
            val_samples = int(len(x_train) * val_split)
            x_val, y_val = x_train[:val_samples], y_train[:val_samples]
            x_train, y_train = x_train[val_samples:], y_train[val_samples:]
        elif args.dataset:
            # 从目录结构加载数据
            train_data, val_data, test_data = preprocessor.load_dataset(args.dataset)
            
            # 解包数据
            x_train, y_train = train_data
            x_val, y_val = val_data
            x_test, y_test = test_data
        else:
            raise ValueError("必须提供数据集路径(--dataset)或FER2013 CSV文件路径(--fer2013)")
        
        print(f"训练样本数: {len(x_train)}")
        print(f"验证样本数: {len(x_val)}")
        print(f"测试样本数: {len(x_test)}")
        
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
    
    # 创建数据生成器
    print("\n创建数据生成器...")
    
    # 获取数据增强配置
    augmentation_config = get_augmentation_config(args.augment_level)
    augment = args.augment_level != 'none'
    
    # 创建ImageDataGenerator实例
    train_datagen = preprocessor.create_data_generator(augment=augment, augmentation_config=augmentation_config)
    val_datagen = preprocessor.create_data_generator(augment=False)
    
    # 使用flow方法创建数据生成器
    train_generator = train_datagen.flow(x_train, y_train, batch_size=args.batch_size)
    val_generator = val_datagen.flow(x_val, y_val, batch_size=args.batch_size, shuffle=False)
    
    if augment:
        print(f"已启用{args.augment_level}级数据增强")
    
    # 加载或创建模型
    print("\n加载/创建模型...")
    try:
        if args.model and os.path.exists(args.model):
            # 从已有模型继续训练
            model = EmotionModel.load_model(args.model)
            print(f"已加载模型: {args.model}")
            
            # 冻结指定层数
            if args.freeze_layers > 0:
                for layer in model.layers[:args.freeze_layers]:
                    layer.trainable = False
                print(f"已冻结前{args.freeze_layers}层")
        else:
            # 创建新模型
            model = EmotionModel.create_model(input_shape=args.input_shape, num_classes=args.num_classes, model_type=args.model_type)
            print(f"已创建{args.model_type}模型")
        
        # 编译模型
        optimizer = get_optimizer(args.optimizer, args.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 打印模型摘要
        model.summary()
        
    except Exception as e:
        print(f"模型加载/创建失败: {e}")
        return
    
    # 准备回调函数
    print("\n准备训练回调函数...")
    callbacks = []
    
    # 检查点回调
    checkpoint_path = os.path.join(args.save_dir, f'{experiment_name}_best_model.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # 早停回调
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # 学习率调整回调
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # CSV日志回调
    csv_logger = CSVLogger(
        os.path.join(args.log_dir, f'{experiment_name}_training.log'),
        append=True,
        separator=';'
    )
    callbacks.append(csv_logger)
    
    # TensorBoard回调
    tensorboard = TensorBoard(
        log_dir=os.path.join(args.log_dir, f'{experiment_name}_logs'),
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    callbacks.append(tensorboard)
    
    # 计算类别权重
    class_weights = None
    if args.class_weight:
        class_weights = calculate_class_weights(y_train)
    
    # 开始训练
    print("\n开始训练...")
    print(f"实验名称: {experiment_name}")
    print(f"训练配置: {args.epochs}轮, 批次大小{args.batch_size}, 学习率{args.learning_rate}")
    
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=len(x_train) // args.batch_size,
            epochs=args.epochs,
            validation_data=val_generator,
            validation_steps=len(x_val) // args.batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # 保存最终模型
        final_model_path = os.path.join(args.save_dir, f'{experiment_name}_final_model.h5')
        model.save(final_model_path)
        print(f"\n最终模型已保存至: {final_model_path}")
        
        # 评估模型
        print("\n评估模型性能...")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        
        # 生成预测结果
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
        
        # 生成分类报告
        print("\n分类报告:")
        report = classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=['生气', '厌恶', '恐惧', '开心', '中性', '伤心', '惊讶'],
            digits=4
        )
        print(report)
        
        # 保存分类报告
        report_path = os.path.join(args.save_dir, f'{experiment_name}_classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"分类报告已保存至: {report_path}")
        
        # 绘制混淆矩阵
        cm_path = os.path.join(args.save_dir, f'{experiment_name}_confusion_matrix.png')
        ModelUtils.plot_confusion_matrix(y_true_classes, y_pred_classes, 
                                        labels=['生气', '厌恶', '恐惧', '开心', '中性', '伤心', '惊讶'],
                                        save_path=cm_path)
        print(f"混淆矩阵已保存至: {cm_path}")
        
        # 绘制训练历史
        history_path = os.path.join(args.save_dir, f'{experiment_name}_training_history.png')
        ModelUtils.plot_training_history(history, save_path=history_path)
        print(f"训练历史图已保存至: {history_path}")
        
        # 提示TensorBoard使用方法
        print("\n训练完成! 可以使用以下命令查看详细训练过程:")
        print(f"tensorboard --logdir {args.log_dir}")
        
    except Exception as e:
        print(f"训练过程出错: {e}")
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")