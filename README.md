# LMS-YOLO

本项目基于[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)进行了多个方面的改进，主要包括网络结构优化和损失函数的改进。

## 主要改进
```
LMS-YOLO\ultralytics\
├── nn\
│   ├── Add\
│   │   └── LCA3.py                # 基于多尺度深度可分离低秩卷积的空间注意力机制
│   └── modules\
│       ├── c2f_UIB.py            # 基于倒置瓶颈结构的模块轻量化方法
│       └── CAWF_.py              # 基于跨层自适应加权的特征融合方法
└── utils\
    └── loss.py                   # 基于尺度自适应的损失优化方法
        └── asiou_loss_improved() # 位于第257-358行
```
### 1. 基于多尺度深度可分离低秩卷积的空间注意力机制

引入了基于多尺度深度可分离低秩卷积的空间注意力机制，这是一个轻量级的局部上下文注意力模块，可以有效捕获目标的局部特征信息。

### 2. 基于倒置瓶颈结构的模块轻量化方法
提出了c2f-UIB模块，用于优化特征提取过程中的信息流动，提高模型对不同尺度目标的检测能力。

### 3. CAWF--基于跨层自适应加权的特征融合方法
设计了CAWF模块，通过高效的双向跨尺度连接与加权特征融合，实现对多尺度特征的充分整合。

### 4. 基于尺度自适应的损失优化方法
提出了ASIoU损失函数，主要特点包括：
- 自适应尺度感知机制，更好地处理不同尺度目标
- 结合了DIoU的中心点距离惩罚项
- 引入Focal机制，优化难样本学习


## 使用方法

### 模型训练
```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8xxx.yaml')  # 使用改进后的配置文件

# 训练模型
results = model.train(
    data='coco.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### 模型推理
```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('best.pt')

# 进行预测
results = model('image.jpg')
```

## 许可证

本项目遵循 AGPL-3.0 开源许可证。详情请参见 [LICENSE](LICENSE) 文件。

## 致谢

感谢 [Ultralytics](https://github.com/ultralytics/ultralytics) 团队开源的YOLOv8项目。
```
