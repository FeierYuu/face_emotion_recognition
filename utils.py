import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
import platform
import sys
import cv2

class FaceUtils:
    """人脸检测和处理的工具类，为unified_emotion_recognition.py提供支持"""
    
    def __init__(self, use_opencv=True):
        # 默认使用OpenCV的Haar级联检测器
        self.use_opencv = use_opencv
        self.has_landmarks = False  # 移除dlib的关键点检测
        self.font = None  # 预加载的字体
        self.font_loaded = False  # 字体加载状态
        
        if self.use_opencv:
            # 尝试加载OpenCV的Haar级联分类器
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                print("使用OpenCV的Haar级联检测器进行人脸检测")
            except Exception as e:
                print(f"加载OpenCV人脸检测器失败: {e}")
                # 作为后备方案，创建一个简单的检测器
                self.face_cascade = None
        else:
            # 这里保留原有的dlib检测代码作为参考，但默认不使用
            try:
                import dlib
                self.face_detector = dlib.get_frontal_face_detector()
                print("使用dlib人脸检测器")
            except ImportError:
                print("未找到dlib模块，自动切换到OpenCV检测器")
                self.use_opencv = True
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 预加载字体
        self._load_font()
        
    def _load_font(self):
        """预加载支持中文显示的字体"""
        if self.font_loaded:
            return
            
        try:
            from PIL import ImageFont
            
            # 尝试加载中文字体
            if platform.system() == 'Darwin':  # macOS
                # 尝试多种常见的中文字体路径
                font_paths = ['/System/Library/Fonts/PingFang.ttc', 
                              '/System/Library/Fonts/SFNS.ttc', 
                              '/Library/Fonts/Arial Unicode.ttf']
                for path in font_paths:
                    if os.path.exists(path):
                        self.font = ImageFont.truetype(path, 20)
                        print(f"预加载字体: {path}")
                        self.font_loaded = True
                        break
            elif platform.system() == 'Windows':
                font_paths = ['C:/Windows/Fonts/simhei.ttf',  # 黑体
                             'C:/Windows/Fonts/msyh.ttc']  # 微软雅黑
                for path in font_paths:
                    if os.path.exists(path):
                        self.font = ImageFont.truetype(path, 20)
                        print(f"预加载字体: {path}")
                        self.font_loaded = True
                        break
            else:  # Linux等其他系统
                font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # 文泉驿微米黑
                if os.path.exists(font_path):
                    self.font = ImageFont.truetype(font_path, 20)
                    print(f"预加载字体: {path}")
                    self.font_loaded = True
            
            if not self.font_loaded:
                print("未找到中文字体，将使用默认字体")
                self.font = ImageFont.load_default()
                self.font_loaded = True
        except Exception as e:
            print(f"加载字体时出错: {e}")
            try:
                from PIL import ImageFont
                self.font = ImageFont.load_default()
                self.font_loaded = True
                print("使用PIL默认字体")
            except:
                print("无法加载任何字体")
    
    def detect_faces(self, image, upsample=1, scale_factor=None, min_neighbors=None):
        """
        检测图像中的人脸
        
        参数:
            image: 输入图像
            upsample: 上采样次数（OpenCV模式下忽略此参数）
            scale_factor: 检测人脸的缩放因子（仅OpenCV模式下有效）
            min_neighbors: 检测人脸的最小邻居数（仅OpenCV模式下有效）
        
        返回:
            人脸矩形列表 [(x1, y1, x2, y2), ...]
        """
        if self.use_opencv or self.face_cascade is not None:
            # 使用OpenCV的Haar级联检测器检测人脸
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸，返回格式为(x, y, w, h)
            # 使用传入的参数或默认值
            scale_factor_val = scale_factor if scale_factor is not None else 1.1
            min_neighbors_val = min_neighbors if min_neighbors is not None else 5
            
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=scale_factor_val, 
                minNeighbors=min_neighbors_val, 
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # 转换为(x1, y1, x2, y2)格式
            face_rects = []
            for (x, y, w, h) in faces:
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                face_rects.append((x1, y1, x2, y2))
        else:
            # 这里保留原有的dlib检测代码，仅在明确指定不使用OpenCV时使用
            try:
                import dlib
                # 使用dlib检测器检测人脸
                faces = self.face_detector(image, upsample)
                
                # 转换为(x1, y1, x2, y2)格式
                face_rects = []
                for face in faces:
                    x1, y1 = face.left(), face.top()
                    x2, y2 = face.right(), face.bottom()
                    face_rects.append((x1, y1, x2, y2))
            except:
                # 如果任何检测方法都失败，返回空列表
                face_rects = []
        
        return face_rects
    
    def preprocess_faces_batch(self, image, face_rects):
        """
        批量预处理多个人脸图像
        
        参数:
            image: 输入图像
            face_rects: 人脸矩形列表 [(x1, y1, x2, y2), ...]
        
        返回:
            预处理后的人脸图像批次
        """
        faces_batch = []
        for (x1, y1, x2, y2) in face_rects:
            try:
                # 裁剪人脸区域
                face = image[y1:y2, x1:x2]
                
                # 转换为灰度图
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                
                # 调整大小为48x48（情绪识别模型的输入大小）
                resized_face = cv2.resize(gray_face, (48, 48))
                
                # 归一化
                normalized_face = resized_face / 255.0
                
                # 添加通道维度
                face_with_channel = np.expand_dims(normalized_face, axis=-1)
                
                # 添加到批次中
                faces_batch.append(face_with_channel)
            except:
                # 如果处理单个人脸失败，跳过
                continue
        
        return np.array(faces_batch)
    
    def detect_faces_opencv(self, image):
        """
        使用OpenCV的Haar级联检测器检测人脸（备用方法，保持向后兼容性）
        
        参数:
            image: 输入图像
        
        返回:
            人脸矩形列表 [(x, y, w, h), ...]
        """
        # 直接调用主要的detect_faces方法，但转换返回格式以保持兼容性
        face_rects = self.detect_faces(image)
        
        # 转换为(x, y, w, h)格式
        opencv_format_faces = []
        for (x1, y1, x2, y2) in face_rects:
            w = x2 - x1
            h = y2 - y1
            opencv_format_faces.append((x1, y1, w, h))
        
        return opencv_format_faces
    
    def draw_emotion_label(self, image, face_rect, emotion, confidence):
        """
        在图像上绘制人脸矩形和情绪标签
        
        参数:
            image: 输入图像
            face_rect: 人脸矩形 (x1, y1, x2, y2)
            emotion: 情绪标签
            confidence: 置信度
        
        返回:
            绘制后的图像
        """
        x1, y1, x2, y2 = face_rect
        
        # 绘制人脸矩形
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制情绪标签
        label = f"{emotion}: {confidence:.2f}%"
        
        # 确保中文能正确显示的核心解决方案
        try:
            # 方法1：使用PIL库来处理中文显示（最可靠的方法）
            from PIL import Image, ImageDraw
            
            # 确保字体已加载
            if not self.font_loaded:
                self._load_font()
                
            # 将OpenCV图像转换为PIL图像
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 使用预加载的字体绘制中文标签
            if self.font:
                draw.text((x1, y1 - 30), label, font=self.font, fill=(0, 255, 0))
            else:
                # 如果没有可用字体，使用PIL默认字体
                draw.text((x1, y1 - 30), label, fill=(0, 255, 0))
            
            # 将PIL图像转换回OpenCV图像
            image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            # 如果使用PIL失败，尝试使用OpenCV的基本方法
            try:
                # 对于非中文系统，可能需要使用Unicode编码
                # 这里使用一种简单的方法，直接显示标签
                cv2.putText(image, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except:
                # 作为最后的后备方案，只显示矩形框不显示文字
                pass
        
        return image
    
    def align_face(self, image, face_rect):
        """
        对齐人脸（简化版本，不依赖dlib关键点检测）
        
        参数:
            image: 输入图像
            face_rect: 人脸矩形
        
        返回:
            对齐后的人脸图像
        """
        try:
            # 直接裁剪人脸区域
            x1, y1, x2, y2 = face_rect
            face_image = image[y1:y2, x1:x2]
            
            # 如果人脸区域有效，进行简单的预处理
            if face_image.size > 0:
                # 调整大小为固定尺寸
                desired_size = 256
                resized_face = cv2.resize(face_image, (desired_size, desired_size))
                return resized_face
            else:
                # 如果裁剪失败，返回空图像
                return np.zeros((desired_size, desired_size, 3), dtype=np.uint8)
        except Exception as e:
            print(f"人脸对齐失败: {e}")
            # 对齐失败时，返回原始裁剪的人脸
            x1, y1, x2, y2 = face_rect
            return image[y1:y2, x1:x2]

class ModelUtils:
    """模型评估和可视化的工具类"""
    
    @staticmethod
    def evaluate_model(model, x_test, y_test, emotion_labels):
        """
        评估模型性能
        
        参数:
            model: 训练好的模型
            x_test: 测试集图像
            y_test: 测试集标签
            emotion_labels: 情绪标签字典
        
        返回:
            评估结果字典
        """
        # 获取预测结果
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # 计算准确率
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        # 生成分类报告
        report = classification_report(y_true_classes, y_pred_classes, 
                                       target_names=[emotion_labels[i] for i in range(len(emotion_labels))])
        
        # 生成混淆矩阵
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        print(f"模型评估结果:")
        print(f"损失: {loss:.4f}")
        print(f"准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(report)
        
        # 绘制混淆矩阵
        ModelUtils.plot_confusion_matrix(cm, [emotion_labels[i] for i in range(len(emotion_labels))])
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    @staticmethod
    def plot_confusion_matrix(cm, class_names):
        """
        绘制混淆矩阵
        
        参数:
            cm: 混淆矩阵
            class_names: 类别名称列表
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # 在混淆矩阵中添加数值标签
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_training_history(history, save_path=None):
        """
        绘制训练历史曲线
        
        参数:
            history: 训练历史对象
            save_path: 保存图像的路径，如果为None则显示图像
        """
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 绘制准确率曲线
        ax1.plot(history.history['accuracy'], label='训练准确率')
        if 'val_accuracy' in history.history:
            ax1.plot(history.history['val_accuracy'], label='验证准确率')
        ax1.set_title('模型准确率')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('准确率')
        ax1.legend()
        
        # 绘制损失曲线
        ax2.plot(history.history['loss'], label='训练损失')
        if 'val_loss' in history.history:
            ax2.plot(history.history['val_loss'], label='验证损失')
        ax2.set_title('模型损失')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('损失')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"训练历史已保存到: {save_path}")
        else:
            plt.show()
        
    @staticmethod
    def download_landmark_model():
        """注意：此方法已废弃，不再需要下载dlib人脸关键点模型
        当前版本使用OpenCV的Haar级联检测器进行人脸检测，无需额外模型文件
        """
        print("提示: 当前版本使用OpenCV的Haar级联检测器，无需下载dlib人脸关键点模型")
        print("dlib模块已不再是必需依赖项，系统已自动切换到纯OpenCV实现")
        return True
    
    @staticmethod
    def get_face_detector_info():
        """获取当前人脸检测器的信息
        
        返回:
            包含检测器信息的字典
        """
        return {
            'detector_type': 'OpenCV Haar Cascade',
            'description': '使用OpenCV内置的Haar级联分类器进行人脸检测',
            'advantages': ['无需额外安装dlib模块', '内置在OpenCV中，无需额外下载模型', 
                         '在大多数平台上都能良好工作', '安装简单，兼容性好'],
            'limitations': ['相比dlib可能准确率稍低', '不支持基于关键点的人脸对齐']
        }