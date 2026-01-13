import pandas as pd

df = pd.read_csv('/home/algo/video_agent_group/qianqian/NeuMachineLearning-main/task4/detection/fovea_localization_train_GT.csv')
print(df.describe())
print(df.head())
