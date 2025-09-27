import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import argparse
import time
from datetime import datetime

# 导入项目模块
from model import EmotionModel
from data_preprocessing import DataPreprocessor
from utils import FaceUtils

class UnifiedEmotionRecognizer:
    """整合了基础和M1优化功能的统一情绪识别器"""
    def __init__(self, model_path='models/emotion_model.h5', use_m1_optimizations=False, 
                 use_tflite=False, verbose=True):
        """
        初始化情绪识别器
        
        参数:
            model_path: 模型路径
            use_m1_optimizations: 是否使用M1优化
            use_tflite: 是否使用TensorFlow Lite模型
            verbose: 是否显示详细信息
        """
        self.model_path = model_path
        self.use_m1_optimizations = use_m1_optimizations
        self.use_tflite = use_tflite
        self.verbose = verbose
        
        # 情绪标签映射 (FER2013数据集标准顺序)
        self.emotion_labels = {
            0: '生气',
            1: '厌恶',
            2: '恐惧',
            3: '开心',
            4: '伤心',
            5: '惊讶',
            6: '中性'
        }
        
        # 初始化组件
        self.model = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        self.face_utils = FaceUtils()
        self.preprocessor = DataPreprocessor()
        
        # 配置TensorFlow
        self._configure_tensorflow()
        
        # 加载模型
        self._load_model()
    
    def _configure_tensorflow(self):
        """配置TensorFlow以提高性能"""
        # 启用内存增长
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if self.verbose:
                    print("GPU内存增长已启用")
            except RuntimeError as e:
                if self.verbose:
                    print(f"启用GPU内存增长失败: {e}")
        
        # 如果启用了M1优化
        if self.use_m1_optimizations:
            if hasattr(tf.config, 'experimental'):
                try:
                    # 尝试配置Metal加速
                    tf.config.experimental.set_visible_devices([], 'GPU')
                    if self.verbose:
                        print("配置为使用M1优化的TensorFlow")
                except Exception as e:
                    if self.verbose:
                        print(f"配置TensorFlow设备失败: {e}")
    
    def _load_model(self):
        """加载模型"""
        # 自动检测文件扩展名并设置相应的加载方式
        if self.model_path.lower().endswith('.tflite'):
            # 如果是.tflite文件，自动使用TFLite模式
            print(f"检测到TFLite模型文件，自动启用TFLite模式")
            self._load_tflite_model(self.model_path)
        elif self.use_tflite:
            # 尝试加载TFLite模型
            # 如果模型路径已经是.tflite结尾，直接使用
            if self.model_path.lower().endswith('.tflite'):
                tflite_path = self.model_path
            else:
                # 否则替换扩展名
                tflite_path = self.model_path.replace('.h5', '.tflite')
            self._load_tflite_model(tflite_path)
        else:
            # 加载Keras模型
            self._load_keras_model(self.model_path)
    
    def _load_keras_model(self, model_path):
        """加载Keras模型"""
        try:
            if self.verbose:
                print(f"正在加载Keras模型: {model_path}")
            self.model = EmotionModel.load_model(model_path)
            # 预热模型
            dummy_input = np.random.rand(1, 48, 48, 1).astype(np.float32)
            self.model.predict(dummy_input, verbose=0)
            if self.verbose:
                print(f"Keras模型加载并预热完成: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print(f"警告: 当前使用的是新创建的模型，可能没有经过充分训练，识别结果可能不准确")
            self.model = EmotionModel.create_model(input_shape=(48, 48, 1), num_classes=7)
    
    def _load_tflite_model(self, tflite_path):
        """加载TensorFlow Lite模型以提高推理速度"""
        try:
            # 验证模型文件是否存在
            if not os.path.exists(tflite_path):
                if self.verbose:
                    print(f"警告: TFLite模型文件不存在: {tflite_path}")
                # 尝试使用默认模型文件路径
                default_tflite_path = 'models/final_emotion_model_optimized.tflite'
                if os.path.exists(default_tflite_path):
                    if self.verbose:
                        print(f"正在使用默认TFLite模型: {default_tflite_path}")
                    tflite_path = default_tflite_path
                else:
                    # 尝试从Keras模型转换
                    keras_path = tflite_path.replace('.tflite', '.h5')
                    if os.path.exists(keras_path):
                        if self.verbose:
                            print(f"从Keras模型转换: {keras_path}")
                        self._convert_to_tflite(keras_path, tflite_path)
                    else:
                        if self.verbose:
                            print("无法找到对应的Keras模型，使用Keras模式")
                        self._load_keras_model(keras_path)
                        return
            
            if self.verbose:
                print(f"正在加载TensorFlow Lite模型: {tflite_path}")
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # 预热模型
            dummy_input = np.random.rand(1, 48, 48, 1).astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
            self.interpreter.get_tensor(self.output_details[0]['index'])
            
            if self.verbose:
                print(f"TensorFlow Lite模型加载并预热完成: {tflite_path}")
        except Exception as e:
            print(f"加载TFLite模型失败: {e}")
            # 回退到Keras模型
            keras_path = tflite_path.replace('.tflite', '.h5')
            print(f"尝试加载对应的Keras模型: {keras_path}")
            self._load_keras_model(keras_path)
    
    def _convert_to_tflite(self, keras_path, tflite_path):
        """将Keras模型转换为TensorFlow Lite模型"""
        try:
            model = EmotionModel.load_model(keras_path)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            # 应用优化
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            # 保存模型
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            if self.verbose:
                print(f"TFLite模型已保存至: {tflite_path}")
        except Exception as e:
            print(f"转换模型失败: {e}")
    
    def predict(self, face_image):
        """预测单个人脸图像的情绪"""
        if self.interpreter is not None:
            # 使用TensorFlow Lite进行推理
            input_data = np.array(face_image, dtype=np.float32)
            if len(input_data.shape) == 3:
                input_data = np.expand_dims(input_data, axis=0)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            return output_data[0]
        else:
            # 使用Keras模型进行推理
            if len(face_image.shape) == 3:
                face_image = np.expand_dims(face_image, axis=0)
            return self.model.predict(face_image, verbose=0)[0]
    
    def predict_emotion(self, face_image, confidence_threshold=0.3):
        """预测人脸情绪，直接返回模型识别结果"""
        # 获取所有情绪的概率
        predictions = self.predict(face_image)
        
        # 找到最高概率的情绪和其索引
        max_prob = np.max(predictions)
        max_index = np.argmax(predictions)
        
        # 只使用简单的置信度阈值过滤
        if max_prob >= confidence_threshold:
            # 直接返回模型识别的情绪结果，不做额外过滤
            return max_index, max_prob
        
        # 如果不满足置信度要求，返回None表示无法确定
        return None, max_prob
    
    def predict_batch(self, faces_batch):
        """批量预测情绪"""
        if len(faces_batch) == 0:
            return []
        
        if self.interpreter is not None:
            # 使用TensorFlow Lite进行推理
            results = []
            for face in faces_batch:
                # TFLite模型通常需要特定的数据类型和正确的维度
                input_data = np.array(face, dtype=np.float32)
                # 确保输入数据是4维的 (批次维度, 高度, 宽度, 通道)
                if len(input_data.shape) == 3:
                    input_data = np.expand_dims(input_data, axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                results.append(output_data[0])
            return np.array(results)
        else:
            # 使用Keras模型进行批量推理
            return self.model.predict(faces_batch, verbose=0)
    
    def recognize_from_image(self, image_path, confidence_threshold=0.3, show=True, save=False):
        """从图像中识别情绪"""
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法加载图像: {image_path}")
            return
        
        print(f"成功加载图像: {image_path}，图像尺寸: {image.shape[:2]}")
        
        # 复制图像用于显示
        display_image = image.copy()
        
        # 检测人脸
        faces = self.face_utils.detect_faces(image)
        
        if not faces:
            print("未检测到人脸")
            # 调整人脸检测参数并再次尝试
            print("尝试使用更宽松的人脸检测参数...")
            faces = self.face_utils.detect_faces(image, scale_factor=1.05, min_neighbors=3)
            
        print(f"检测到 {len(faces)} 个人脸")
        
        # 处理每个人脸
        emotion_detected = False
        for (x1, y1, x2, y2) in faces:
            print(f"人脸位置: ({x1}, {y1}, {x2}, {y2})，尺寸: {x2-x1}x{y2-y1}")
            # 从图像中裁剪人脸区域
            face_region = image[y1:y2, x1:x2]
            # 预处理人脸
            face_image = self.preprocessor.preprocess_face(face_region)
            
            # 使用新的预测方法，增加额外的过滤逻辑
            emotion_index, confidence = self.predict_emotion(face_image, confidence_threshold)
            confidence_percent = confidence * 100
            
            print(f"人脸情绪预测: 最高置信度 = {confidence_percent:.1f}%")
            
            # 如果情绪识别结果有效，显示结果
            if emotion_index is not None:
                emotion = self.emotion_labels.get(emotion_index, f'未知({emotion_index})')
                display_image = self.face_utils.draw_emotion_label(display_image, (x1, y1, x2, y2), emotion, confidence_percent)
                print(f"识别结果: {emotion} ({confidence_percent:.1f}%)")
                emotion_detected = True
            else:
                # 只显示人脸框，表示无法确定情绪
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                print(f"未识别出情绪 (置信度低于阈值: {confidence_threshold})")
        
        if not emotion_detected:
            print(f"提示: 尝试使用更低的置信度阈值(如0.3)来提高识别率")
            print(f"示例命令: python unified_emotion_recognition.py --image {image_path} --confidence 0.3")
        
        # 显示结果
        if show:
            cv2.imshow("情绪识别结果", display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # 保存结果
        if save:
            save_path = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(save_path, display_image)
            print(f"结果已保存至: {save_path}")
    
    def recognize_from_camera(self, resolution=(640, 480), skip_frames=0, confidence_threshold=0.3, show_fps=True):
        """
        从摄像头实时识别人脸情绪
        
        参数:
            resolution: 摄像头分辨率 (width, height)
            skip_frames: 帧跳过数量，0表示处理每一帧
            confidence_threshold: 情绪识别的置信度阈值
            show_fps: 是否显示帧率
        """
        # 初始化人脸跟踪变量
        tracked_faces = []  # 保存前一帧的人脸位置
        track_counter = 0  # 跟踪计数器
        track_max_frames = 5  # 最多跟踪5帧
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        # 设置摄像头分辨率以提高性能
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        if self.verbose:
            print(f"\n开始实时情绪识别 (分辨率: {resolution[0]}x{resolution[1]}, 跳过帧数: {skip_frames}, 阈值: {confidence_threshold})" +
                  f"{', M1优化已启用' if self.use_m1_optimizations else ''}" +
                  f"{', TFLite已启用' if self.use_tflite else ''}")
            print("按ESC键退出")
        
        # 用于计算帧率
        prev_time = 0
        frame_count = 0
        
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 帧跳过策略（仅在启用M1优化时使用）
            if self.use_m1_optimizations and skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                # 显示上一帧的结果
                cv2.imshow("实时情绪识别", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            
            # 复制帧用于显示
            display_frame = frame.copy()
            
            # 计算并显示帧率
            if show_fps:
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
                prev_time = current_time
                # 在图像上显示帧率
                cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 初始化faces为空列表，确保变量始终有定义
            faces = []
            
            # 优先使用更严格的参数进行人脸检测
            if self.use_m1_optimizations:
                faces = self.face_utils.detect_faces(frame, scale_factor=1.1, min_neighbors=5)
            else:
                faces = self.face_utils.detect_faces(frame, scale_factor=1.1, min_neighbors=5)
            
            # 如果没有检测到人脸，尝试使用更宽松的参数
            if not faces:
                faces = self.face_utils.detect_faces(frame, scale_factor=1.05, min_neighbors=3)
            
            # 人脸跟踪逻辑
            current_confidence_threshold = confidence_threshold
            if faces:
                tracked_faces = faces
                track_counter = 0
            elif tracked_faces and track_counter < track_max_frames:
                # 如果没有检测到新人脸但有跟踪记录，使用跟踪位置
                faces = tracked_faces
                track_counter += 1
                # 在跟踪模式下使用更低的置信度阈值
                current_confidence_threshold = max(0.2, confidence_threshold - 0.1)
            
            # 确保即使在跟踪模式下也能显示人脸框
            # 先绘制所有人脸框（不依赖情绪识别结果）
            for (x1, y1, x2, y2) in faces:
                # 绘制人脸框
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 在M1优化模式下使用批量处理
            if self.use_m1_optimizations:
                faces_batch = self.face_utils.preprocess_faces_batch(frame, faces)
                
                if len(faces_batch) > 0:
                    # 批量预测情绪
                    predictions = self.predict_batch(faces_batch)
                    
                    # 处理预测结果
                    for i, (x1, y1, x2, y2) in enumerate(faces):
                        if i >= len(predictions):
                            continue
                        
                        prediction = predictions[i]
                        # 找到最高概率的情绪和其索引
                        max_prob = np.max(prediction)
                        max_index = np.argmax(prediction)
                        
                        # 使用简单的置信度阈值过滤
                        valid_emotion = False
                        if max_prob >= current_confidence_threshold:
                            # 直接使用模型识别的情绪结果，不做额外过滤
                            valid_emotion = True
                        
                        # 如果情绪识别结果有效，显示结果
                        if valid_emotion:
                            emotion = self.emotion_labels.get(max_index, f'未知({max_index})')
                            confidence_percent = max_prob * 100
                            display_frame = self.face_utils.draw_emotion_label(display_frame, (x1, y1, x2, y2), emotion, confidence_percent)
            else:
                # 标准处理方式
                for (x1, y1, x2, y2) in faces:
                    # 从帧中裁剪人脸区域
                    face_region = frame[y1:y2, x1:x2]
                    # 预处理人脸
                    face_image = self.preprocessor.preprocess_face(face_region)
                    
                    # 使用新的预测方法，增加额外的过滤逻辑
                    emotion_index, confidence = self.predict_emotion(face_image, current_confidence_threshold)
                    confidence_percent = confidence * 100
                    
                    # 如果情绪识别结果有效，显示结果
                    if emotion_index is not None:
                        emotion = self.emotion_labels.get(emotion_index, f'未知({emotion_index})')
                        display_frame = self.face_utils.draw_emotion_label(display_frame, (x1, y1, x2, y2), emotion, confidence_percent)
            
            # 显示结果
            cv2.imshow("实时情绪识别", display_frame)
            
            # 按ESC键退出
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        if self.verbose:
            print("情绪识别已停止")
    
    def recognize_from_video(self, video_path, confidence_threshold=0.5, show_fps=True):
        """从视频文件中识别情绪"""
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.verbose:
            print(f"\n开始处理视频: {video_path}")
            print(f"视频信息: {width}x{height}, {fps:.2f} FPS, {frame_count} 帧")
            print("按ESC键退出")
        
        # 用于计算处理帧率
        prev_time = 0
        
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                break
            
            # 复制帧用于显示
            display_frame = frame.copy()
            
            # 计算并显示处理帧率
            if show_fps:
                current_time = time.time()
                process_fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
                prev_time = current_time
                # 在图像上显示处理帧率
                cv2.putText(display_frame, f"处理FPS: {process_fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 检测人脸
            faces = self.face_utils.detect_faces(frame)
            
            # 处理人脸
            for (x1, y1, x2, y2) in faces:
                # 从帧中裁剪人脸区域
                face_region = frame[y1:y2, x1:x2]
                # 预处理人脸
                face_image = self.preprocessor.preprocess_face(face_region)
                
                # 使用新的预测方法，增加额外的过滤逻辑
                emotion_index, confidence = self.predict_emotion(face_image, confidence_threshold)
                confidence_percent = confidence * 100
                
                # 如果情绪识别结果有效，显示结果
                if emotion_index is not None:
                    emotion = self.emotion_labels.get(emotion_index, f'未知({emotion_index})')
                    display_frame = self.face_utils.draw_emotion_label(display_frame, (x1, y1, x2, y2), emotion, confidence_percent)
            
            # 显示结果
            cv2.imshow("视频情绪识别", display_frame)
            
            # 按ESC键退出
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        if self.verbose:
            print("视频情绪识别已完成")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='统一的面部情绪识别系统')
    
    # 输入类型参数
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--image', type=str, help='要处理的图像路径')
    input_group.add_argument('--video', type=str, help='要处理的视频路径')
    input_group.add_argument('--camera', action='store_true', default=True, help='使用摄像头进行实时识别（默认）')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='models/final_emotion_model_optimized.tflite',
                      help='预训练模型的路径 (默认使用优化的TFLite模型)')
    parser.add_argument('--tflite_model', type=str, default=None,
                      help='直接指定TensorFlow Lite模型的路径（优先级高于model参数）')
    parser.add_argument('--confidence', type=float, default=0.3,
                      help='情绪识别的置信度阈值，建议值: 0.3-0.5')
    
    # 优化参数
    parser.add_argument('--use_m1_optimizations', action='store_true',
                      help='启用M1芯片优化')
    parser.add_argument('--use_tflite', action='store_true',
                      help='使用TensorFlow Lite模型进行更快的推理')
    
    # 摄像头和显示参数
    parser.add_argument('--resolution', type=int, nargs=2, default=(640, 480),
                      help='摄像头分辨率 (宽度 高度)')
    parser.add_argument('--skip_frames', type=int, default=0,
                      help='帧跳过数量，0表示处理每一帧（仅在M1优化模式下有效）')
    parser.add_argument('--show_fps', action='store_true', default=True,
                      help='显示帧率')
    parser.add_argument('--save_result', action='store_true',
                      help='保存处理结果（仅对图像输入有效）')
    
    # 调试参数
    parser.add_argument('--verbose', action='store_true', default=True,
                      help='显示详细信息')
    
    parser.add_argument('--use-keras', action='store_true',
                      help='使用Keras模型而不是TFLite模型')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置模型路径和TFLite模式
    model_path = args.model
    # 检查是否强制使用Keras模型
    use_keras = args.use_keras
    if use_keras:
        model_path = model_path.replace('.tflite', '.h5')
        print(f'强制使用Keras模型: {model_path}')
    # 自动检测TFLite模型并启用相应模式
    if model_path and model_path.endswith('.tflite'):
        args.use_tflite = True  # 自动启用TFLite
    # 如果指定了tflite_model参数，则优先使用它
    if args.tflite_model:
        model_path = args.tflite_model
        args.use_tflite = True  # 自动启用TFLite
    
    # 创建情绪识别器
    recognizer = UnifiedEmotionRecognizer(
        model_path=model_path,
        use_m1_optimizations=args.use_m1_optimizations,
        use_tflite=args.use_tflite,
        verbose=args.verbose
    )
    
    # 根据输入类型执行相应的识别功能
    if args.image:
        # 图像情绪识别
        recognizer.recognize_from_image(args.image, 
                                      confidence_threshold=args.confidence, 
                                      show=True, 
                                      save=args.save_result)
    elif args.video:
        # 视频情绪识别
        recognizer.recognize_from_video(args.video, 
                                      confidence_threshold=args.confidence,
                                      show_fps=args.show_fps)
    else:
        # 实时摄像头情绪识别
        recognizer.recognize_from_camera(
            resolution=args.resolution,
            skip_frames=args.skip_frames,
            confidence_threshold=args.confidence,
            show_fps=args.show_fps
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")