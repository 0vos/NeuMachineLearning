# SAM下载
cd /home/algo/video_agent_group/qianqian
source qqenv/bin/activate

python - <<'PY'
from ultralytics.utils.downloads import attempt_download_asset
print(attempt_download_asset("sam2.1_s.pt"))
print(attempt_download_asset("sam_b.pt"))
PY

# SAM细化后生成CSV
python NeuMachineLearning-main/task5/task5.py refine_sam \
  --split test \
  --unet_checkpoint NeuMachineLearning-main/task5/outputs_vgg16_unetpp/fold_5/best.pth \
  --sam_variant sam2 \
  --sam_weights sam2.1_l.pt \
  --combine and \
  --prior_thr 0.03 \
  --postprocess \
  --fundus_margin_px 12 \
  --border_px 14 \
  --frame_px 6 \
  --min_area_blob 120 \
  --min_area_line 25 \
  --elong_thr 10.0 \
  --out_dir NeuMachineLearning-main/task5/segmentation/image

# optional: extra denoise + thinning (keeps line-like vessels, reduces thick masks)
python NeuMachineLearning-main/task5/task5_postprocess_v2.py \
  --image_dir NeuMachineLearning-main/task5/segmentation/test/image \
  --mask_dir NeuMachineLearning-main/task5/segmentation/image \
  --out_dir NeuMachineLearning-main/task5/segmentation/image \
  --img_size 512 \
  --fundus_margin_px 12 \
  --frame_px 6 \
  --border_px 14 \
  --min_area_blob 120 \
  --min_area_line 25 \
  --elong_thr 10.0 \
  --thin \
  --dilate_px 1 \
  --cap_radius_px 6

python NeuMachineLearning-main/task5/task5.py to_csv

# SAM123对比（都用U-net训练，然后分别用sam123输出Dice图对比）
cd /home/algo/video_agent_group/qianqian
source qqenv/bin/activate

python NeuMachineLearning-main/task5/task5.py compare_sam \
  --data_root NeuMachineLearning-main/task5/segmentation \
  --out_dir NeuMachineLearning-main/task5/compare_sam \
  --folds 5 \
  --epochs 6 \
  --batch_size 2 \
  --sam1_weights sam_b.pt \
  --sam2_weights sam2.1_s.pt \
  --sam3_weights sam3_*.pt \
  --sam3_text "blood vessel"