# 贡献指南

感谢您考虑为面部情绪识别项目做出贡献！我们欢迎各种形式的贡献，包括但不限于：

- 🐛 报告Bug
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码修复
- 🧪 添加测试用例

## 开发环境设置

### 1. Fork 和克隆仓库

```bash
# Fork 本仓库到您的GitHub账户
# 然后克隆您的fork
git clone https://github.com/yourusername/face_emotion_recognition.git
cd face_emotion_recognition

# 添加上游仓库
git remote add upstream https://github.com/originalowner/face_emotion_recognition.git
```

### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 贡献流程

### 1. 创建分支

```bash
# 从main分支创建新分支
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

### 2. 进行更改

- 编写清晰的代码
- 添加必要的注释
- 确保代码符合项目风格
- 添加或更新相关测试

### 3. 提交更改

```bash
# 添加更改的文件
git add .

# 提交更改（使用清晰的提交信息）
git commit -m "Add: 添加新功能描述"
# 或
git commit -m "Fix: 修复某个问题"
```

### 4. 推送并创建Pull Request

```bash
# 推送到您的fork
git push origin feature/your-feature-name

# 在GitHub上创建Pull Request
```

## 代码规范

### Python代码风格

- 使用PEP 8代码风格
- 函数和变量使用snake_case
- 类名使用PascalCase
- 常量使用UPPER_CASE
- 行长度限制在120字符以内

### 提交信息规范

使用以下格式：

```
类型: 简短描述

详细描述（可选）

- 类型: Add, Fix, Update, Remove, Refactor, Docs, Test
- 描述: 简洁明了地描述更改内容
```

示例：
```
Add: 添加M1芯片优化支持

- 实现TensorFlow Metal加速
- 添加M1专用配置选项
- 优化内存使用
```

## 测试

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_emotion_recognition.py

# 生成覆盖率报告
python -m pytest --cov=. tests/
```

### 添加新测试

- 为新功能添加相应的测试用例
- 确保测试覆盖率达到80%以上
- 使用描述性的测试名称

## 文档

### 更新文档

- 更新README.md中的相关部分
- 为新功能添加使用示例
- 更新API文档（如有）

### 代码注释

- 为复杂逻辑添加注释
- 使用docstring描述函数和类
- 保持注释的时效性

## 问题报告

### 报告Bug

使用GitHub Issues报告问题时，请包含：

1. **问题描述**：清晰描述问题
2. **重现步骤**：详细的重现步骤
3. **预期行为**：描述您期望的行为
4. **实际行为**：描述实际发生的情况
5. **环境信息**：
   - 操作系统
   - Python版本
   - 依赖包版本
6. **错误日志**：相关的错误信息或日志

### 功能请求

提出新功能时，请包含：

1. **功能描述**：详细描述您想要的功能
2. **使用场景**：说明为什么需要这个功能
3. **实现建议**：如果有的话，提供实现建议
4. **替代方案**：描述您考虑过的其他方案

## 代码审查

### 审查清单

提交代码前，请确保：

- [ ] 代码符合项目风格
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 提交信息清晰
- [ ] 没有破坏现有功能
- [ ] 性能影响已考虑

### 审查流程

1. 自动检查通过（CI/CD）
2. 至少一位维护者审查
3. 所有讨论解决后合并

## 联系方式

如有问题，可以通过以下方式联系：

- GitHub Issues
- 邮件：your-email@example.com
- 讨论区：GitHub Discussions

## 许可证

通过贡献代码，您同意您的贡献将在MIT许可证下发布。

感谢您的贡献！🎉
