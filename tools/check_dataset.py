import os
import sys
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.general import check_file, colorstr

def check_yaml_labels(data_file):
    """检查yaml配置和标签文件"""
    with open(data_file) as f:
        data = yaml.safe_load(f)
    
    # 检查基本配置
    assert 'train' in data, '未找到训练集路径'
    assert 'val' in data, '未找到验证集路径'
    assert 'nc' in data, '未指定类别数量'
    assert 'names' in data, '未指定类别名称'
    
    prefix = colorstr('检查数据集: ')
    problems = []
    stats = {'missing_labels': 0, 'wrong_format': 0, 'empty_labels': 0}
    
    # 检查训练集
    train_path = Path(data['train'])
    if train_path.is_dir():
        train_files = list(train_path.rglob('*.jpg')) + list(train_path.rglob('*.png'))
        print(f"{prefix}检查训练集: {len(train_files)} 个图片")
        
        for img_file in tqdm(train_files, desc='检查训练集标签'):
            label_file = str(img_file).replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            if not os.path.exists(label_file):
                stats['missing_labels'] += 1
                problems.append(f"缺失标签文件: {label_file}")
                continue
            
            # 检查标签格式
            try:
                labels = np.loadtxt(label_file)
                if labels.size == 0:
                    stats['empty_labels'] += 1
                    problems.append(f"空标签文件: {label_file}")
                if len(labels.shape) == 1:
                    labels = labels.reshape(-1, 5)
                if not (labels[:, 1:] <= 1).all():
                    stats['wrong_format'] += 1
                    problems.append(f"标签坐标未归一化: {label_file}")
            except Exception as e:
                stats['wrong_format'] += 1
                problems.append(f"标签格式错误 {label_file}: {str(e)}")
    
    # 打印统计信息
    print(f"\n{prefix}检查结果:")
    print(f"总图片数量: {len(train_files)}")
    print(f"缺失标签文件: {stats['missing_labels']}")
    print(f"空标签文件: {stats['empty_labels']}")
    print(f"格式错误: {stats['wrong_format']}")
    
    if problems:
        print("\n前10个问题:")
        for p in problems[:10]:
            print(p)

    return stats['missing_labels'] == 0 and stats['wrong_format'] == 0

def main(opt):
    check_yaml_labels(opt.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset.yaml path')
    opt = parser.parse_args()
    main(opt)