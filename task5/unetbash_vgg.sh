cd /home/algo/video_agent_group/qianqian
source qqenv/bin/activate

# train (v2: UNet++ + VGG encoder, 5-fold CV)
# Notes:
# - VGG encoders are heavier; start with batch_size=1.
# - If OOM: reduce --img_size to 448 or 384.
python NeuMachineLearning-main/task5/task5_v2.py train_cv \
  --data_root NeuMachineLearning-main/task5/segmentation \
  --out_dir NeuMachineLearning-main/task5/outputs_vgg16_unetpp \
  --folds 5 \
  --model unetplusplus \
  --encoder vgg16_bn \
  --encoder_weights imagenet \
  --loss focal_dice \
  --img_size 512 \
  --batch_size 1 \
  --epochs 100 \
  --lr 3e-4 \
  --min_lr 1e-6 \
  --weight_decay 1e-4 \
  --seed 42 \
  --num_workers 2

# Optional alt: vgg19_bn
# python NeuMachineLearning-main/task5/task5_v2.py train_cv \
#   --data_root NeuMachineLearning-main/task5/segmentation \
#   --out_dir NeuMachineLearning-main/task5/outputs_vgg19_unetpp \
#   --folds 5 \
#   --model unetplusplus \
#   --encoder vgg19_bn \
#   --encoder_weights imagenet \
#   --loss focal_dice \
#   --img_size 512 \
#   --batch_size 1 \
#   --epochs 100 \
#   --lr 3e-4 \
#   --min_lr 1e-6 \
#   --weight_decay 1e-4 \
#   --seed 42 \
#   --num_workers 2

# inference (v2: ensemble + TTA)
cd /home/algo/video_agent_group/qianqian
source qqenv/bin/activate

python NeuMachineLearning-main/task5/task5_v2.py predict_ensemble \
  --data_root NeuMachineLearning-main/task5/segmentation \
  --split test \
  --ckpt_root NeuMachineLearning-main/task5/outputs_vgg16_unetpp \
  --folds 5 \
  --out_dir NeuMachineLearning-main/task5/segmentation/image \
  --threshold 0.45 \
  --tta

# get csv
cd /home/algo/video_agent_group/qianqian
source qqenv/bin/activate

python NeuMachineLearning-main/task5/task5_v2.py to_csv
