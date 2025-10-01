#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多语言情绪识别测试脚本
演示中文、英文、俄语三种语言的显示效果
"""
import os
import sys
from unified_emotion_recognition import UnifiedEmotionRecognizer

def test_multilingual_support():
    """测试多语言支持功能"""
    print("🌍 多语言情绪识别系统测试")
    print("=" * 50)
    
    # 测试图像路径
    test_image = "images/90.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        return
    
    # 测试三种语言
    languages = [
        ('zh', '中文'),
        ('en', 'English'),
        ('ru', 'Русский')
    ]
    
    for lang_code, lang_name in languages:
        print(f"\n🔤 测试语言: {lang_name} ({lang_code})")
        print("-" * 30)
        
        try:
            # 创建识别器
            recognizer = UnifiedEmotionRecognizer(
                model_path='models/final_emotion_model_optimized.tflite',
                use_tflite=True,
                verbose=False,
                language=lang_code
            )
            
            # 显示当前语言的情绪标签
            print(f"当前语言标签映射:")
            for idx, label in recognizer.emotion_labels.items():
                print(f"  {idx}: {label}")
            
            # 测试动态语言切换
            print(f"\n🔄 测试动态语言切换...")
            recognizer.set_language('en')
            print(f"切换到英文: {recognizer.emotion_labels[3]}")  # 开心
            
            recognizer.set_language('ru')
            print(f"切换到俄语: {recognizer.emotion_labels[3]}")  # 开心
            
            recognizer.set_language(lang_code)
            print(f"切换回{lang_name}: {recognizer.emotion_labels[3]}")  # 开心
            
            print(f"✅ {lang_name} 语言支持正常")
            
        except Exception as e:
            print(f"❌ {lang_name} 语言测试失败: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 多语言功能测试完成！")
    print("\n📖 使用方法:")
    print("python unified_emotion_recognition.py --image images/90.jpg --language en")
    print("python unified_emotion_recognition.py --image images/90.jpg --language ru")
    print("python unified_emotion_recognition.py --image images/90.jpg --language zh")

def show_emotion_mappings():
    """显示所有语言的情绪映射"""
    print("\n📚 完整情绪标签映射表:")
    print("=" * 60)
    
    # 创建三种语言的映射
    mappings = {
        'zh': {0: '生气', 1: '厌恶', 2: '恐惧', 3: '开心', 4: '伤心', 5: '惊讶', 6: '中性'},
        'en': {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'},
        'ru': {0: 'Злой', 1: 'Отвращение', 2: 'Страх', 3: 'Счастливый', 4: 'Грустный', 5: 'Удивленный', 6: 'Нейтральный'}
    }
    
    print(f"{'索引':<4} {'中文':<8} {'English':<12} {'Русский':<15}")
    print("-" * 60)
    
    for i in range(7):
        zh_label = mappings['zh'][i]
        en_label = mappings['en'][i]
        ru_label = mappings['ru'][i]
        print(f"{i:<4} {zh_label:<8} {en_label:<12} {ru_label:<15}")

if __name__ == "__main__":
    show_emotion_mappings()
    test_multilingual_support()
