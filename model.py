import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
    Input, GlobalAveragePooling2D, Add, Activation
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2

class EmotionModel:
    """面部情绪识别模型类"""
    
    @staticmethod
    def create_model(input_shape=(48, 48, 1), num_classes=7, model_type='advanced'):
        """
        创建情绪识别的CNN模型
        
        参数:
            input_shape: 输入图像形状，默认为(48, 48, 1)（灰度图）
            num_classes: 情绪类别数量，默认为7
            model_type: 模型类型，可选'basic'或'advanced'，默认为'advanced'
        
        返回:
            构建好的Keras模型
        """
        if model_type == 'basic':
            # 基础模型（原模型）
            return EmotionModel._create_basic_model(input_shape, num_classes)
        else:
            # 高级模型（增强版）
            return EmotionModel._create_advanced_model(input_shape, num_classes)
    
    @staticmethod
    def _create_basic_model(input_shape=(48, 48, 1), num_classes=7):
        """创建基础版本的模型"""
        model = Sequential()
        
        # 第一卷积块
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # 第二卷积块
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # 第三卷积块
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # 全连接层
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # 输出层
        model.add(Dense(num_classes, activation='softmax'))
        
        # 编译模型
        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        return model
    
    @staticmethod
    def _create_advanced_model(input_shape=(48, 48, 1), num_classes=7):
        """创建高级版本的模型，具有更好的特征提取能力"""
        inputs = Input(shape=input_shape)
        
        # 第一卷积块 - 浅层特征提取
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.3)(x)  # 增加Dropout比例
        
        # 第二卷积块 - 中级特征提取
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.3)(x)
        
        # 第三卷积块 - 深层特征提取
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.3)(x)
        
        # 第四卷积块 - 更丰富的特征提取
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        
        # 使用全局平均池化代替部分全连接层，减少过拟合
        x = GlobalAveragePooling2D()(x)
        
        # 全连接层 - 分类
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        # 输出层
        outputs = Dense(num_classes, activation='softmax')(x)
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型 - 使用RMSprop优化器通常对图像任务效果更好
        optimizer = RMSprop(learning_rate=0.0001)
        model.compile(optimizer=optimizer, 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        return model
    
    @staticmethod
    def get_emotion_labels():
        """返回情绪标签映射"""
        return {
            0: '生气',
            1: '厌恶',
            2: '恐惧',
            3: '开心',
            4: '伤心',
            5: '惊讶',
            6: '中性'
        }
        
    @staticmethod
    def save_model(model, filepath='models/emotion_model.h5'):
        """保存模型到文件"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model.save(filepath)
        print(f"模型已保存到: {filepath}")
        
    @staticmethod
    def load_model(filepath='models/emotion_model.h5'):
        """从文件加载模型"""
        from tensorflow.keras.models import load_model
        try:
            model = load_model(filepath)
            print(f"模型已从 {filepath} 加载")
            return model
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("使用新创建的高级模型")
            return EmotionModel.create_model(model_type='advanced')
    
    @staticmethod
    def create_custom_model(input_shape=(48, 48, 1), num_classes=7, optimizer_type='rmsprop', learning_rate=0.0001):
        """
        创建自定义配置的模型
        
        参数:
            input_shape: 输入图像形状
            num_classes: 类别数量
            optimizer_type: 优化器类型，可选'adam'或'rmsprop'
            learning_rate: 学习率
        
        返回:
            构建好的Keras模型
        """
        model = EmotionModel._create_advanced_model(input_shape, num_classes)
        
        # 根据选择的优化器类型编译模型
        if optimizer_type.lower() == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        else:
            optimizer = RMSprop(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model