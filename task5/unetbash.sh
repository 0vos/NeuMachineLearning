cd /home/algo/video_agent_group/qianqian
source qqenv/bin/activate

# train (v2: 5-fold CV + UNet-family baseline, non-ResNet encoder)
python NeuMachineLearning-main/task5/task5_v2.py train_cv \
  --data_root NeuMachineLearning-main/task5/segmentation \
  --out_dir NeuMachineLearning-main/task5/outputs_v2 \
  --folds 5 \
  --model unetplusplus \
  --encoder efficientnet-b4 \
  --encoder_weights imagenet \
  --loss focal_dice \
  --img_size 512 \
  --batch_size 2 \
  --epochs 80 \
  --lr 3e-4 \
  --min_lr 1e-6 \
  --weight_decay 1e-4 \
  --seed 42 \
  --num_workers 2

# inference (v2: ensemble + TTA)
cd /home/algo/video_agent_group/qianqian
source qqenv/bin/activate

python NeuMachineLearning-main/task5/task5_v2.py predict_ensemble \
  --data_root NeuMachineLearning-main/task5/segmentation \
  --split test \
  --ckpt_root NeuMachineLearning-main/task5/outputs_v2 \
  --folds 5 \
  --out_dir NeuMachineLearning-main/task5/segmentation/image \
  --threshold 0.45 \
  --tta

# get csv
cd /home/algo/video_agent_group/qianqian
source qqenv/bin/activate

python NeuMachineLearning-main/task5/task5_v2.py to_csv