from utils.general import strip_optimizer

# 只需指定你的权重文件路径即可
# strip_optimizer('runs/train/exp/weights/best.pt')
strip_optimizer('runs/train/yolov75/weights/best.pt', 'runs/train/yolov75/weights/best-stripped.pt')
