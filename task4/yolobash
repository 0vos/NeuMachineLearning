python NeuMachineLearning-main/task4/task4_yolo.py predict \
  --weights NeuMachineLearning-main/task4/runs/task4_yolo/weights/best.pt \
  --imgsz 640 \
  --conf 0.001 \
  --iou 0.5 \
  --device 5 \
  --debug \
  --vis-dir NeuMachineLearning-main/task4/runs/task4_yolo/vis_test \
  --out-csv NeuMachineLearning-main/task4/submission_task4.csv

# new train
python NeuMachineLearning-main/task4/task4_yolo.py train-refine \
  --model yolo11n.pt \
  --imgsz 640 \
  --epochs 200 \
  --batch 32 \
  --device 5 \
  --crop-scale 4.0 \
  --jitter 8 \
  --coarse-weights NeuMachineLearning-main/task4/runs/task4_yolo/weights/best.pt \
  --coarse-imgsz 640 \
  --coarse-conf 0.001 \
  --coarse-device 5 \
  --project NeuMachineLearning-main/task4/runs \
  --name task4_yolo_refine \
  --out-root NeuMachineLearning-main/task4/yolo_dataset \
  --clean

# new predict
python NeuMachineLearning-main/task4/task4_yolo.py predict \
    --weights NeuMachineLearning-main/task4/runs/task4_yolo/weights/best.pt \
    --imgsz 640 \
    --conf 0.001 \
    --iou 0.5 \
    --device 5 \
    --debug \
    --vis-dir NeuMachineLearning-main/task4/runs/task4_yolo/vis_test \
    --refine-weights NeuMachineLearning-main/task4/runs/task4_yolo_refine/weights/best.pt \
    --refine-imgsz 640 \
    --refine-conf 0.001 \
    --refine-crop-scale 4.0 \
    --refine-accept-conf 0.35 \
    --refine-max-shift 0.2 \
    --out-csv NeuMachineLearning-main/task4/submission_task4.csv

# eval
python NeuMachineLearning-main/task4/task4_yolo.py eval \
  --weights NeuMachineLearning-main/task4/runs/task4_yolo/weights/best.pt \
  --imgsz 640 --conf 0.001 --iou 0.5 --device 5 \
  --seed 42 --val-ratio 0.2 --topk 15 \
  --report-csv NeuMachineLearning-main/task4/runs/eval_coarse.csv \
  --vis-dir NeuMachineLearning-main/task4/runs/eval_vis_coarse

# refine eval
python NeuMachineLearning-main/task4/task4_yolo.py eval \
  --weights NeuMachineLearning-main/task4/runs/task4_yolo/weights/best.pt \
  --refine-weights NeuMachineLearning-main/task4/runs/task4_yolo_refine/weights/best.pt \
  --imgsz 640 --conf 0.001 --iou 0.5 --device 5 \
  --refine-imgsz 640 --refine-conf 0.001 --refine-crop-scale 4.0 \
  --refine-accept-conf 0.25 --refine-max-shift 0.35 \
  --seed 42 --val-ratio 0.2 --topk 15 \
  --report-csv NeuMachineLearning-main/task4/runs/eval_refine.csv \
  --vis-dir NeuMachineLearning-main/task4/runs/eval_vis_refine

# yoloZ
# yoloN strong (yolo11x) + 上采样增强 (repeat-train)
python NeuMachineLearning-main/task4/task4_yoloN.py train \
  --preset strong \
  --model yolo11x.pt \
  --repeat-train 3 \
  --device 5 \
  --name task4_yolo_11x_r3 \
  --project NeuMachineLearning-main/task4/runs \
  --out-root NeuMachineLearning-main/task4/yolo_dataset \
  --clean

# yoloN 复合推理：优先 yolo11x；若 yolo11x 低置信度，切换到 yolo11n；
# 同时从 yolo11x 的 topK 候选框中，挑最接近 yolo11n 预测中心的那个，减少 outlier
python NeuMachineLearning-main/task4/task4_yoloN.py predict \
  --weights NeuMachineLearning-main/task4/runs/task4_yolo_11x_optuna_auto_retry/weights/best.pt \
  --imgsz 1280 \
  --conf 0.001 \
  --iou 0.5 \
  --device 6 \
  --fallback center \
  --aux-weights NeuMachineLearning-main/task4/runs/task4_yolo/weights/best.pt \
  --aux-switch-conf 0.08 \
  --aux-imgsz 640 \
  --lowconf-thr 0.08 \
  --lowconf-mode topk_mean \
  --lowconf-topk 5 \
  --debug \
  --vis-dir NeuMachineLearning-main/task4/runs/task4_yolo_11x_optuna_auto_retry/vis_test \
  --out-csv NeuMachineLearning-main/task4/submission_task4.csv

# (可选) 多卡训练版本：注意 batch 必须 >= GPU 数
python NeuMachineLearning-main/task4/task4_yoloN.py train \
  --preset strong \
  --model yolo11x.pt \
  --repeat-train 3 \
  --device 0,1,2,3,4,5,6,7 \
  --batch 64 \
  --name task4_yolo_11x_r3_ddp \
  --project NeuMachineLearning-main/task4/runs \
  --out-root NeuMachineLearning-main/task4/yolo_dataset \
  --clean


# -------------------------
# Optuna 自动调参（HPO）
# -------------------------
# 建议：先用小模型 (yolo11n/s) 快速搜索增强+学习率等超参；
# 然后把 best_params 固定下来，再用 yolo11x 认真训练。
#
# 1) 安装 optuna
# pip install optuna
#
# 2) 运行调参（会训练 trials 次，并用 val split 的坐标 MSE 作为目标）
python NeuMachineLearning-main/task4/task4_yoloN.py tune \
  --preset strong \
  --model yolo11s.pt \
  --device 5 \
  --epochs 80 \
  --batch 16 \
  --repeat-train 3 \
  --trials 40 \
  --storage sqlite:////tmp/task4_optuna.db \
  --study-name task4_yolo_tune_s \
  --base-name task4_optuna_s \
  --project NeuMachineLearning-main/task4/runs \
  --out-root NeuMachineLearning-main/task4/yolo_dataset_optuna \
  --out-json /tmp/task4_optuna_best.json \
  --final-model yolo11x.pt \
  --final-name task4_yolo_11x_optuna_auto \
  --final-device 5 \
  --final-epochs 400 \
  --final-batch 64 \
  --final-imgsz 1280 \
  --final-repeat-train 3 \
  --final-project NeuMachineLearning-main/task4/runs \
  --final-out-root NeuMachineLearning-main/task4/yolo_dataset \
  --final-clean

# 3) 用 eval 验证某个 weights 的 val MSE（也可以看 worst-10 图）
python NeuMachineLearning-main/task4/task4_yoloN.py eval \
  --weights NeuMachineLearning-main/task4/runs/task4_yolo_11x_optuna_auto_retry/weights/best.pt \
  --imgsz 1024 \
  --device 6 \
  --out-report /tmp/task4_val_report.csv \
  --debug

# 4) 上面这条 tune 命令会自动：
#    - 输出 best_params 到 /tmp/task4_optuna_best.json
#    - 立刻用 best_params 启动最终 yolo11x 训练（final_weights 会打印出来）

# 5) 如果 final 训练因为 CUDA OOM 中断：trial 的结果不会丢。
#    你可以“只跑 final、不重跑 trials”，把 batch/imgsz 调小即可：
python NeuMachineLearning-main/task4/task4_yoloN.py tune \
  --preset strong \
  --model yolo11s.pt \
  --device 5 \
  --epochs 80 \
  --batch 16 \
  --repeat-train 3 \
  --trials 0 \
  --storage sqlite:////tmp/task4_optuna.db \
  --study-name task4_yolo_tune_s \
  --base-name task4_optuna_s \
  --project NeuMachineLearning-main/task4/runs \
  --out-root NeuMachineLearning-main/task4/yolo_dataset_optuna \
  --out-json /tmp/task4_optuna_best.json \
  --final-model yolo11x.pt \
  --final-name task4_yolo_11x_optuna_auto_retry \
  --final-device 5 \
  --final-epochs 400 \
  --final-batch 16 \
  --final-imgsz 1024 \
  --final-repeat-train 3 \
  --final-project NeuMachineLearning-main/task4/runs \
  --final-out-root NeuMachineLearning-main/task4/yolo_dataset \
  --final-clean


# -------------------------
# Optuna trials (0..39) 集成推理
# -------------------------
# 说明：把 task4_optuna_s_t0000..0039 的 best.pt 逐个跑一遍，
#      对每张 test 图把中心点做聚合（默认 median 更抗 outlier），输出 submission。
python NeuMachineLearning-main/task4/task4_yoloN.py ensemble \
  --ensemble-root NeuMachineLearning-main/task4/runs \
  --ensemble-prefix task4_optuna_s_t \
  --ensemble-start 0 \
  --ensemble-end 39 \
  --ensemble-suffix weights/best.pt \
  --ensemble-mode median \
  --ensemble-min-conf 0.02 \
  --imgsz 1024 \
  --conf 0.001 \
  --iou 0.5 \
  --device 6 \
  --fallback center \
  --debug \
  --out-csv NeuMachineLearning-main/task4/submission_task4_ens_trials0_39.csv

# 0..39 trials + (yolo11x optuna final) + (baseline yolo11n best) 一起集成
python NeuMachineLearning-main/task4/task4_yoloN.py ensemble \
  --ensemble-root NeuMachineLearning-main/task4/runs \
  --ensemble-prefix task4_optuna_s_t \
  --ensemble-start 0 \
  --ensemble-end 39 \
  --ensemble-suffix weights/best.pt \
  --ensemble-weights \
    NeuMachineLearning-main/task4/runs/task4_yolo/weights/best.pt \
  --ensemble-mode median \
  --ensemble-min-conf 0.02 \
  --imgsz 1024 \
  --conf 0.001 \
  --iou 0.5 \
  --device 5 \
  --fallback center \
  --debug \
  --out-csv NeuMachineLearning-main/task4/submission_task4_ens_trials0_39_plus11x_plusyolon.csv