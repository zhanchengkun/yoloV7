import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载模型
# model = torch.hub.load('WongKinYiu/yolov7', 'custom', path='yolov7.pt', source='github')
model = torch.load('yolov7.pt') 
# 读取图像
img = cv2.imread('path_to_your_image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 进行推理
results = model(img_rgb)

# 处理结果
results.print()  # 打印检测结果
detections = results.xyxy[0].numpy()  # 获取检测结果

# 绘制检测框
for *box, conf, cls in detections:
    label = f'{model.names[int(cls)]} {conf:.2f}'
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    cv2.putText(img, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 显示结果
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
