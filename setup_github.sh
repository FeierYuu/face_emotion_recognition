#!/bin/bash

# GitHub仓库设置脚本
# 用于初始化Git仓库并准备上传到GitHub

echo "🚀 开始设置GitHub仓库..."

# 检查是否已存在.git目录
if [ -d ".git" ]; then
    echo "⚠️  检测到已存在Git仓库，跳过初始化"
else
    echo "📁 初始化Git仓库..."
    git init
fi

# 添加所有文件
echo "📝 添加文件到Git..."
git add .

# 创建初始提交
echo "💾 创建初始提交..."
git commit -m "Initial commit: 面部情绪识别系统

- 实现基于深度学习的面部情绪识别
- 支持实时摄像头、图像和视频识别
- 优化支持Apple M1/M2芯片
- 包含完整的训练和测试功能
- 提供详细的使用文档和贡献指南"

# 显示状态
echo "📊 Git状态："
git status

echo ""
echo "✅ 本地Git仓库设置完成！"
echo ""
echo "📋 下一步操作："
echo "1. 在GitHub上创建新仓库"
echo "2. 添加远程仓库："
echo "   git remote add origin https://github.com/yourusername/face_emotion_recognition.git"
echo "3. 推送代码："
echo "   git push -u origin main"
echo ""
echo "🔗 或者使用GitHub CLI："
echo "   gh repo create face_emotion_recognition --public --source=. --remote=origin --push"
echo ""
echo "📚 更多信息请查看 README.md 和 CONTRIBUTING.md"
