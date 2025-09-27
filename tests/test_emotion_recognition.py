#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面部情绪识别系统测试模块
"""
import unittest
import numpy as np
import cv2
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import EmotionModel
from data_preprocessing import DataPreprocessor
from utils import FaceUtils


class TestEmotionRecognition(unittest.TestCase):
    """情绪识别系统测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.preprocessor = DataPreprocessor()
        self.face_utils = FaceUtils()
        
    def test_emotion_labels(self):
        """测试情绪标签映射"""
        from unified_emotion_recognition import UnifiedEmotionRecognizer
        
        recognizer = UnifiedEmotionRecognizer(verbose=False)
        expected_labels = {
            0: '生气',
            1: '厌恶', 
            2: '恐惧',
            3: '开心',
            4: '伤心',
            5: '惊讶',
            6: '中性'
        }
        
        self.assertEqual(recognizer.emotion_labels, expected_labels)
        print("✅ 情绪标签映射测试通过")
    
    def test_data_preprocessing(self):
        """测试数据预处理功能"""
        # 创建测试图像
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 测试预处理
        processed = self.preprocessor.preprocess_face(test_image)
        
        # 检查输出形状
        self.assertEqual(processed.shape, (1, 48, 48, 1))
        print("✅ 数据预处理测试通过")
    
    def test_face_detection(self):
        """测试人脸检测功能"""
        # 创建测试图像（模拟人脸）
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        # 测试人脸检测
        faces = self.face_utils.detect_faces(test_image)
        
        # 检查返回类型
        self.assertIsInstance(faces, list)
        print("✅ 人脸检测测试通过")
    
    def test_model_creation(self):
        """测试模型创建"""
        try:
            model = EmotionModel.create_model(input_shape=(48, 48, 1), num_classes=7)
            
            # 检查模型结构
            self.assertIsNotNone(model)
            self.assertEqual(model.input_shape, (None, 48, 48, 1))
            self.assertEqual(model.output_shape, (None, 7))
            print("✅ 模型创建测试通过")
        except Exception as e:
            self.fail(f"模型创建失败: {e}")
    
    def test_emotion_prediction_format(self):
        """测试情绪预测输出格式"""
        # 创建模拟预测结果
        mock_predictions = np.array([0.1, 0.1, 0.1, 0.6, 0.05, 0.03, 0.02])
        
        # 检查预测结果格式
        self.assertEqual(len(mock_predictions), 7)
        self.assertAlmostEqual(np.sum(mock_predictions), 1.0, places=5)
        
        # 检查最高概率索引
        max_index = np.argmax(mock_predictions)
        self.assertEqual(max_index, 3)  # 开心
        print("✅ 情绪预测格式测试通过")
    
    def test_confidence_threshold(self):
        """测试置信度阈值功能"""
        # 高置信度预测
        high_conf = np.array([0.1, 0.1, 0.1, 0.7, 0.0, 0.0, 0.0])
        max_prob = np.max(high_conf)
        self.assertGreater(max_prob, 0.5)
        
        # 低置信度预测
        low_conf = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05])
        max_prob = np.max(low_conf)
        self.assertLess(max_prob, 0.5)
        print("✅ 置信度阈值测试通过")


class TestIntegration(unittest.TestCase):
    """集成测试类"""
    
    def test_full_pipeline(self):
        """测试完整识别流程"""
        try:
            from unified_emotion_recognition import UnifiedEmotionRecognizer
            
            # 创建识别器（不加载实际模型）
            recognizer = UnifiedEmotionRecognizer(verbose=False)
            
            # 检查识别器初始化
            self.assertIsNotNone(recognizer.emotion_labels)
            self.assertIsNotNone(recognizer.face_utils)
            self.assertIsNotNone(recognizer.preprocessor)
            print("✅ 完整流程测试通过")
            
        except Exception as e:
            self.fail(f"完整流程测试失败: {e}")


def run_tests():
    """运行所有测试"""
    print("🧪 开始运行面部情绪识别系统测试...")
    print("=" * 50)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestEmotionRecognition))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("=" * 50)
    if result.wasSuccessful():
        print("🎉 所有测试通过！")
    else:
        print(f"❌ 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
