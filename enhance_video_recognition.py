#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

"""
视频识别优化工具：解决人脸框时有时无和情绪识别不准确的问题

此脚本提供以下功能：
1. 优化人脸检测参数，提高检测稳定性
2. 调整情绪识别阈值，减少中性被误判为惊讶的情况
3. 添加人脸跟踪逻辑，保持检测框连续性
4. 允许切换回原始Keras模型

使用方法：
  python enhance_video_recognition.py --optimize-face-detection
  python enhance_video_recognition.py --adjust-neutral-threshold
  python enhance_video_recognition.py --enable-face-tracking
"""

# 定义文件路径
UNIFIED_FILE_PATH = 'unified_emotion_recognition.py'


def read_file(file_path):
    """读取文件内容"""
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


def write_file(file_path, lines):
    """写入文件内容"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def optimize_face_detection(lines):
    """优化人脸检测参数，添加备用检测策略"""
    modified = False
    for i, line in enumerate(lines):
        if 'faces = self.face_utils.detect_faces(frame, scale_factor=1.1, min_neighbors=5)' in line:
            # 在严格参数检测后添加备用检测逻辑
            insert_index = i + 1
            # 插入备用检测逻辑
            lines.insert(insert_index, "            # 如果没有检测到人脸，尝试使用更宽松的参数\n")
            lines.insert(insert_index + 1, "            if not faces:\n")
            lines.insert(insert_index + 2, "                faces = self.face_utils.detect_faces(frame, scale_factor=1.05, min_neighbors=3)\n")
            modified = True
            print("已添加备用人脸检测策略，在严格参数检测不到时自动使用宽松参数")
            break
    return lines, modified


def adjust_neutral_threshold(lines):
    """调整中性情绪的识别阈值，降低被误判为惊讶的概率"""
    modified = False
    for i, line in enumerate(lines):
        if 'adjusted_threshold = min(confidence_threshold, 0.4)  # 中性情绪最低阈值为0.4' in line:
            # 降低中性情绪的阈值要求
            lines[i] = lines[i].replace('0.4', '0.3')
            lines[i] = lines[i].replace('中性情绪最低阈值为0.4', '中性情绪最低阈值为0.3（降低以减少误判）')
            modified = True
            print("已将中性情绪阈值从0.4降低到0.3，有助于减少面无表情被误判为惊讶的情况")
        elif 'adjusted_threshold = min(confidence_threshold, 0.4)  # 中性情绪最低阈值为0.4' in line:
            # 检查批量处理逻辑中的阈值
            lines[i] = lines[i].replace('0.4', '0.3')
            lines[i] = lines[i].replace('中性情绪最低阈值为0.4', '中性情绪最低阈值为0.3（降低以减少误判）')
            modified = True
    return lines, modified


def enable_face_tracking(lines):
    """添加人脸跟踪逻辑，保持检测框的连续性"""
    modified = False
    # 1. 在recognize_from_camera方法开头添加跟踪变量初始化
    for i, line in enumerate(lines):
        if '    def recognize_from_camera(self, resolution=(640, 480), skip_frames=0, confidence_threshold=0.3, show_fps=True):' in line:
            # 找到方法定义，在方法体内部添加变量初始化
            method_body_start = -1
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == '"""':
                    method_body_start = j + 1
                    break
            
            if method_body_start != -1:
                # 在摄像头初始化前添加跟踪变量
                insert_pos = method_body_start
                while lines[insert_pos].strip() != '# 打开摄像头':
                    insert_pos += 1
                
                lines.insert(insert_pos, "        # 初始化人脸跟踪变量\n")
                lines.insert(insert_pos + 1, "        tracked_faces = []  # 保存前一帧的人脸位置\n")
                lines.insert(insert_pos + 2, "        track_counter = 0  # 跟踪计数器\n")
                lines.insert(insert_pos + 3, "        track_max_frames = 5  # 最多跟踪5帧\n")
                
                # 2. 在人脸检测后添加跟踪逻辑
                for j in range(i, len(lines)):
                    if '            # 处理人脸' in lines[j] and '            if faces:' in lines[j + 1]:
                        # 修改现有代码，添加跟踪逻辑
                        original_lines = lines[j+1:]
                        lines[j+1:j+2] = [
                            "            # 如果检测到新的人脸，更新跟踪列表\n",
                            "            if faces:\n",
                            "                tracked_faces = faces\n",
                            "                track_counter = 0\n",
                            "            elif tracked_faces and track_counter < track_max_frames:\n",
                            "                # 如果没有检测到新人脸但有跟踪记录，使用跟踪位置\n",
                            "                faces = tracked_faces\n",
                            "                track_counter += 1\n",
                            "                # 在跟踪模式下使用更低的置信度阈值\n",
                            "                tracking_confidence_threshold = max(0.2, confidence_threshold - 0.1)\n"
                        ]
                        break
                
                # 3. 在预测情绪时使用跟踪阈值（如果适用）
                for j in range(i, len(lines)):
                    if '                    emotion_index, confidence = self.predict_emotion(face_image, confidence_threshold)' in lines[j]:
                        lines[j] = lines[j].replace('confidence_threshold', 'tracking_confidence_threshold if locals().get(\'tracking_confidence_threshold\') else confidence_threshold')
                        break
                
                # 4. 在批量处理中添加类似逻辑
                for j in range(i, len(lines)):
                    if '                                # 应用与predict_emotion相同的过滤逻辑' in lines[j]:
                        # 找到批量处理中的阈值检查
                        for k in range(j, len(lines)):
                            if '                            if max_prob >= confidence_threshold:' in lines[k]:
                                lines[k] = lines[k].replace('confidence_threshold', 'tracking_confidence_threshold if locals().get(\'tracking_confidence_threshold\') else confidence_threshold')
                                break
                        break
                
                modified = True
                print("已启用人脸跟踪功能，可以在短时间内保持人脸框的连续性")
                break
    return lines, modified


def enable_keras_model_switch(lines):
    """添加Keras模型切换选项"""
    modified = False
    # 1. 在parse_args函数中添加--use-keras参数
    for i, line in enumerate(lines):
        if 'def parse_args():' in line:
            # 找到参数解析函数
            in_parser = False
            for j in range(i, len(lines)):
                if 'parser = argparse.ArgumentParser(' in lines[j]:
                    in_parser = True
                elif in_parser and 'return parser.parse_args()' in lines[j]:
                    # 在返回前添加新参数
                    lines.insert(j, "    parser.add_argument('--use-keras', action='store_true',\n")
                    lines.insert(j + 1, "                      help='使用Keras模型而不是TFLite模型')\n")
                    modified = True
                    break
    
    # 2. 在main函数中处理--use-keras参数
    if modified:
        for i, line in enumerate(lines):
            if '# 自动检测TFLite模型并启用相应模式' in line:
                # 在自动检测前添加--use-keras的处理
                insert_pos = i
                lines.insert(insert_pos, "    # 检查是否强制使用Keras模型\n")
                lines.insert(insert_pos + 1, "    use_keras = args.use_keras\n")
                lines.insert(insert_pos + 2, "    if use_keras:\n")
                lines.insert(insert_pos + 3, "        model_path = model_path.replace('.tflite', '.h5')\n")
                lines.insert(insert_pos + 4, "        print(f'强制使用Keras模型: {model_path}')\n")
                break
        print("已添加--use-keras参数，允许用户切换回原始Keras模型")
    return lines, modified


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='情绪识别系统视频优化工具')
    parser.add_argument('--optimize-face-detection', action='store_true',
                      help='优化人脸检测参数，添加备用检测策略')
    parser.add_argument('--adjust-neutral-threshold', action='store_true',
                      help='降低中性情绪阈值，减少被误判为惊讶的情况')
    parser.add_argument('--enable-face-tracking', action='store_true',
                      help='启用人脸跟踪功能，保持检测框连续性')
    parser.add_argument('--enable-keras-switch', action='store_true',
                      help='添加Keras模型切换选项')
    parser.add_argument('--all', action='store_true',
                      help='应用所有优化')
    args = parser.parse_args()
    
    # 读取文件内容
    lines = read_file(UNIFIED_FILE_PATH)
    if lines is None:
        return
    
    # 创建备份文件
    backup_path = f"{UNIFIED_FILE_PATH}.video_backup"
    if not os.path.exists(backup_path):
        write_file(backup_path, lines)
        print(f"已创建视频识别优化备份: {backup_path}")
    
    # 根据参数进行优化
    any_modified = False
    
    if args.all or args.optimize_face_detection:
        lines, modified = optimize_face_detection(lines)
        any_modified = any_modified or modified
    
    if args.all or args.adjust_neutral_threshold:
        lines, modified = adjust_neutral_threshold(lines)
        any_modified = any_modified or modified
    
    if args.all or args.enable_face_tracking:
        lines, modified = enable_face_tracking(lines)
        any_modified = any_modified or modified
    
    if args.all or args.enable_keras_switch:
        lines, modified = enable_keras_model_switch(lines)
        any_modified = any_modified or modified
    
    # 保存修改后的文件
    if any_modified:
        write_file(UNIFIED_FILE_PATH, lines)
        print(f"视频识别优化已保存到: {UNIFIED_FILE_PATH}")
    else:
        print("没有应用任何优化，请指定优化选项")
    
    # 显示使用建议
    print("\n使用建议：")
    print("1. 应用所有优化（推荐）:")
    print("   python enhance_video_recognition.py --all")
    print("2. 仅优化人脸检测稳定性:")
    print("   python enhance_video_recognition.py --optimize-face-detection")
    print("3. 修复中性表情被误判为惊讶的问题:")
    print("   python enhance_video_recognition.py --adjust-neutral-threshold")
    print("\n优化后运行命令:")
    print("   python unified_emotion_recognition.py --confidence 0.4")
    print("   # 使用Keras模型（如果TFLite效果不佳）")
    print("   python unified_emotion_recognition.py --use-keras")


if __name__ == "__main__":
    main()