import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """数据预处理类，用于处理面部情绪识别的数据"""
    
    @staticmethod
    def load_image(image_path, target_size=(48, 48), grayscale=True):
        """
        加载并预处理单张图像
        
        参数:
            image_path: 图像路径
            target_size: 目标尺寸
            grayscale: 是否转换为灰度图
        
        返回:
            预处理后的图像数组
        """
        try:
            # 加载图像
            image = cv2.imread(image_path)
            
            if image is None:
                raise FileNotFoundError(f"无法加载图像: {image_path}")
            
            # 转换为灰度图
            if grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 调整大小
            image = cv2.resize(image, target_size)
            
            # 归一化
            image = image / 255.0
            
            # 扩展维度以匹配模型输入
            if grayscale:
                image = np.expand_dims(image, axis=-1)
            
            return image
            
        except Exception as e:
            print(f"加载图像时出错: {e}")
            return None
    
    @staticmethod
    def load_dataset(dataset_path, target_size=(48, 48), grayscale=True, test_size=0.2):
        """
        从目录结构加载数据集
        假设目录结构为: dataset_path/情绪类别/图像文件
        
        参数:
            dataset_path: 数据集路径
            target_size: 目标尺寸
            grayscale: 是否转换为灰度图
            test_size: 测试集比例
        
        返回:
            (x_train, y_train), (x_test, y_test): 训练集和测试集
        """
        images = []
        labels = []
        emotion_labels = DataPreprocessor.get_emotion_folders(dataset_path)
        
        for label, emotion in enumerate(emotion_labels):
            emotion_folder = os.path.join(dataset_path, emotion)
            if not os.path.isdir(emotion_folder):
                continue
                
            for image_file in os.listdir(emotion_folder):
                if image_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(emotion_folder, image_file)
                    image = DataPreprocessor.load_image(image_path, target_size, grayscale)
                    
                    if image is not None:
                        images.append(image)
                        labels.append(label)
        
        # 转换为numpy数组
        x = np.array(images)
        y = tf.keras.utils.to_categorical(labels, num_classes=len(emotion_labels))
        
        # 分割训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        
        print(f"数据集加载完成: {len(x_train)} 个训练样本, {len(x_test)} 个测试样本")
        print(f"情绪类别: {emotion_labels}")
        
        return (x_train, y_train), (x_test, y_test)
    
    @staticmethod
    def get_emotion_folders(dataset_path):
        """获取数据集中的情绪类别文件夹"""
        emotion_folders = []
        if os.path.exists(dataset_path):
            for folder in os.listdir(dataset_path):
                folder_path = os.path.join(dataset_path, folder)
                if os.path.isdir(folder_path):
                    emotion_folders.append(folder)
        return sorted(emotion_folders)
    
    @staticmethod
    def create_data_generator(augment=True, augmentation_config=None):
        """
        创建数据生成器，用于数据增强和批量处理
        
        参数:
            augment: 是否进行数据增强
            augmentation_config: 可选的数据增强配置字典
        
        返回:
            ImageDataGenerator实例
        """
        if augment:
            if augmentation_config:
                # 使用提供的配置
                datagen = ImageDataGenerator(**augmentation_config)
            else:
                # 默认增强配置
                datagen = ImageDataGenerator(
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='nearest'
                )
        else:
            datagen = ImageDataGenerator()
        
        return datagen
    
    @staticmethod
    def preprocess_face(face_image, target_size=(48, 48)):
        """
        预处理检测到的人脸图像
        
        参数:
            face_image: 人脸图像
            target_size: 目标尺寸
        
        返回:
            预处理后的人脸图像
        """
        # 转换为灰度图
        face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # 调整大小
        face_resized = cv2.resize(face_gray, target_size)
        
        # 归一化
        face_normalized = face_resized / 255.0
        
        # 扩展维度以匹配模型输入
        face_processed = np.expand_dims(face_normalized, axis=-1)
        face_processed = np.expand_dims(face_processed, axis=0)
        
        return face_processed
    
    @staticmethod
    def load_fer2013_dataset(csv_path, target_size=(48, 48)):
        """
        加载FER2013数据集（CSV格式）
        
        参数:
            csv_path: CSV文件路径
            target_size: 目标尺寸
        
        返回:
            (x_train, y_train), (x_test, y_test): 训练集和测试集
        """
        import pandas as pd
        
        # 加载CSV文件
        data = pd.read_csv(csv_path)
        
        # 分割训练集和测试集
        train_data = data[data['Usage'] == 'Training']
        test_data = data[data['Usage'] != 'Training']
        
        # 处理训练数据
        x_train = []
        y_train = []
        for idx, row in train_data.iterrows():
            pixels = np.array(row['pixels'].split(' '), dtype=np.uint8)
            image = pixels.reshape(48, 48)
            image = cv2.resize(image, target_size)
            x_train.append(image / 255.0)
            y_train.append(row['emotion'])
        
        # 处理测试数据
        x_test = []
        y_test = []
        for idx, row in test_data.iterrows():
            pixels = np.array(row['pixels'].split(' '), dtype=np.uint8)
            image = pixels.reshape(48, 48)
            image = cv2.resize(image, target_size)
            x_test.append(image / 255.0)
            y_test.append(row['emotion'])
        
        # 转换为numpy数组
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        
        # 扩展维度
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        
        # 转换标签为one-hot编码
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=7)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=7)
        
        print(f"FER2013数据集加载完成: {len(x_train)} 个训练样本, {len(x_test)} 个测试样本")
        
        return (x_train, y_train), (x_test, y_test)