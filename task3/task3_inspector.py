import os

def count_images_in_categories(paths):
    """
    统计给定路径下train和test文件夹中每个类别中的图像数量。

    参数:
    paths (list): 包含路径的列表，每个路径应指向包含train和test文件夹的目录。

    返回:
    dict: 包含统计结果的字典，格式为 {path: {'train': {category: count}, 'test': {category: count or total_count}}}
    """
    results = {}
    
    for path in paths:
        results[path] = {'train': {}, 'test': {}}
        
        # 处理train文件夹
        train_path = os.path.join(path, 'train')
        if os.path.exists(train_path):
            for category in os.listdir(train_path):
                category_path = os.path.join(train_path, category)
                if os.path.isdir(category_path):
                    count = len([f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
                    results[path]['train'][category] = count
        
        # 处理test文件夹
        test_path = os.path.join(path, 'test')
        if os.path.exists(test_path):
            # 检查test下是否有子文件夹（类别）
            subdirs = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
            if subdirs:
                # 如果有子文件夹，按类别统计
                for category in subdirs:
                    category_path = os.path.join(test_path, category)
                    count = len([f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
                    results[path]['test'][category] = count
            else:
                # 如果没有子文件夹，统计所有图像文件
                count = len([f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
                results[path]['test']['total'] = count
    
    return results

# 示例使用
paths = [
    '/home/algo/video_agent_group/qianqian/NeuMachineLearning-main/task3/neu-image-emotion-classification/fer_data/fer_data/train',
    '/home/algo/video_agent_group/qianqian/NeuMachineLearning-main/task3/neu-image-emotion-classification/fer_data',
    '/home/algo/video_agent_group/qianqian/NeuMachineLearning-main/task3/neu-image-emotion-classification/fer_data/fer_data'
]

# 注意：第一个路径是train文件夹本身，不是包含train的目录，所以需要调整
# 用户的路径是 fer_data/fer_data/train，所以第一个是train文件夹
# 第二个是fer_data，第三个是fer_data/fer_data

# 调整路径
adjusted_paths = [
    '/home/algo/video_agent_group/qianqian/NeuMachineLearning-main/task3/neu-image-emotion-classification/fer_data/fer_data',  # 包含train和test
    '/home/algo/video_agent_group/qianqian/NeuMachineLearning-main/task3/neu-image-emotion-classification/fer_data',  # 可能没有train/test
    '/home/algo/video_agent_group/qianqian/NeuMachineLearning-main/task3/neu-image-emotion-classification/fer_data/fer_data'  # 重复
]

# 实际上，fer_data下只有fer_data文件夹，所以第二个路径没有train/test
# 第三个是重复的

# 主要路径是 fer_data/fer_data

results = count_images_in_categories(adjusted_paths)
for path, data in results.items():
    print(f"Path: {path}")
    print("Train:")
    for cat, cnt in data['train'].items():
        print(f"  {cat}: {cnt}")
    print("Test:")
    for cat, cnt in data['test'].items():
        print(f"  {cat}: {cnt}")
    print()
