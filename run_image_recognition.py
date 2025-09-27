#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的图像情绪识别脚本
用于帮助用户快速测试图像情绪识别功能
"""
import os
import sys
import time
from unified_emotion_recognition import UnifiedEmotionRecognizer


def main():
    print("=== 图像情绪识别测试工具 ===")
    print("这个工具将帮助您快速测试图像情绪识别功能")
    
    # 获取图像路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("请输入图像路径 (默认为 'images/90.jpg'): ") or "images/90.jpg"
    
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件 '{image_path}' 不存在!")
        print("请检查文件路径是否正确")
        return
    
    # 获取置信度阈值
    try:
        confidence_threshold = float(input("请输入置信度阈值 (默认: 0.3，建议: 0.3-0.5): ") or "0.3")
    except ValueError:
        print("输入无效，使用默认值 0.3")
        confidence_threshold = 0.3
    
    # 获取模型路径
    model_path = input("请输入模型路径 (默认为 'models/final_emotion_model_optimized.tflite'): ") or "models/final_emotion_model_optimized.tflite"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"警告: 模型文件 '{model_path}' 不存在!")
        print("将尝试使用默认模型...")
        model_path = None
    
    # 创建识别器实例
    print("\n正在初始化情绪识别器...")
    try:
        if model_path and model_path.endswith('.tflite'):
            recognizer = UnifiedEmotionRecognizer(model_path=model_path, use_tflite=True)
        else:
            recognizer = UnifiedEmotionRecognizer(model_path=model_path)
        print("情绪识别器初始化成功!")
    except Exception as e:
        print(f"初始化失败: {e}")
        print("请确保模型文件正确且已安装所有依赖")
        return
    
    # 执行情绪识别
    print(f"\n正在处理图像: {image_path}...")
    start_time = time.time()
    try:
        # 使用较低的置信度阈值以提高识别率
        recognizer.recognize_from_image(
            image_path=image_path,
            confidence_threshold=confidence_threshold,
            show=True,
            save=True
        )
        print(f"\n识别完成! 耗时: {time.time() - start_time:.2f}秒")
    except Exception as e:
        print(f"识别过程出错: {e}")
        print("请检查错误信息并尝试解决问题")
    
    # 提供额外帮助信息
    print("\n=== 提示信息 ===")
    print("如果未检测到人脸或情绪，请尝试:")
    print("1. 使用更低的置信度阈值 (如0.2)")
    print("2. 确保图像中人脸清晰可见")
    print("3. 尝试使用其他图像")
    print("\n更多帮助: 参考项目文档或运行 'python unified_emotion_recognition.py --help'")


if __name__ == "__main__":
    main()