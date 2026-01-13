import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw


@dataclass
class Paths:
    root: Path
    det_root: Path
    train_dir: Path
    test_dir: Path
    train_xml_dir: Path
    sample_sub: Path
    gt_csv: Path


def get_paths() -> Paths:
    root = Path(__file__).resolve().parent
    det_root = root / "detection"
    return Paths(
        root=root,
        det_root=det_root,
        train_dir=det_root / "train",
        test_dir=det_root / "test",
        train_xml_dir=det_root / "train_location",
        sample_sub=det_root / "sample_submission.csv",
        gt_csv=det_root / "fovea_localization_train_GT.csv",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _parse_voc_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    xml_w = int(size_node.findtext("width")) if size_node is not None else None
    xml_h = int(size_node.findtext("height")) if size_node is not None else None

    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name is not None and name.lower() != "fovea":
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))
        boxes.append([xmin, ymin, xmax, ymax])

    return boxes, (xml_w, xml_h) if (xml_w and xml_h) else None


def _scale_boxes_to_image(boxes, xml_hw, img_wh):
    if xml_hw is None:
        return boxes
    xml_w, xml_h = xml_hw
    img_w, img_h = img_wh
    if xml_w <= 0 or xml_h <= 0:
        return boxes
    if xml_w == img_w and xml_h == img_h:
        return boxes
    sx = img_w / float(xml_w)
    sy = img_h / float(xml_h)
    scaled = []
    for xmin, ymin, xmax, ymax in boxes:
        scaled.append([xmin * sx, ymin * sy, xmax * sx, ymax * sy])
    return scaled


def _xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    # clamp
    xmin = max(0.0, min(xmin, img_w - 1))
    xmax = max(0.0, min(xmax, img_w - 1))
    ymin = max(0.0, min(ymin, img_h - 1))
    ymax = max(0.0, min(ymax, img_h - 1))

    bw = max(0.0, xmax - xmin)
    bh = max(0.0, ymax - ymin)
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0

    # normalize
    return cx / img_w, cy / img_h, bw / img_w, bh / img_h


def prepare_yolo_dataset(args) -> Path:
    """Create a Ultralytics YOLO dataset folder with images/labels and a dataset yaml."""
    paths = get_paths()
    out_root = Path(args.out_root)
    if out_root.exists() and args.clean:
        shutil.rmtree(out_root)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    ids = list(range(1, 81))
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    val_size = max(1, int(len(ids) * args.val_ratio))
    val_ids = set(ids[:val_size])

    for image_id in range(1, 81):
        img_src = paths.train_dir / f"{image_id:04d}.jpg"
        xml_path = paths.train_xml_dir / f"{image_id:04d}.xml"
        if not img_src.exists() or not xml_path.exists():
            continue

        with Image.open(img_src) as im:
            img_w, img_h = im.size

        boxes, xml_hw = _parse_voc_xml(xml_path)
        boxes = _scale_boxes_to_image(boxes, xml_hw, (img_w, img_h))

        split = "val" if image_id in val_ids else "train"
        img_dst = out_root / "images" / split / img_src.name
        lbl_dst = out_root / "labels" / split / f"{img_src.stem}.txt"

        shutil.copy2(img_src, img_dst)

        # One class: 0=fovea
        lines = []
        for xmin, ymin, xmax, ymax in boxes:
            cx, cy, bw, bh = _xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            # YOLO expects 0..1
            cx = min(1.0, max(0.0, cx))
            cy = min(1.0, max(0.0, cy))
            bw = min(1.0, max(0.0, bw))
            bh = min(1.0, max(0.0, bh))
            lines.append(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")

        lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_path = out_root / "task4_fovea.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: fovea",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return yaml_path


def train_yolo(args) -> Path:
    from ultralytics import YOLO

    yaml_path = prepare_yolo_dataset(args)
    model = YOLO(args.model)

    # Ultralytics supports multi-GPU by passing device list string: "0,1,2,3,4,5,6,7"
    device = args.device

    # IMPORTANT: In Ultralytics DDP, `batch` is the TOTAL batch size, then split across GPUs.
    # If total batch < num_gpus, per-rank batch becomes 0 and training crashes.
    n_gpus = 0
    if isinstance(device, str) and device not in {"cpu", "-1"}:
        if "," in device:
            n_gpus = len([d for d in device.split(",") if d.strip() != ""])
        else:
            # single gpu like "0"
            try:
                int(device)
                n_gpus = 1
            except Exception:
                n_gpus = 0

    total_batch = int(args.batch)
    if n_gpus >= 2 and total_batch < n_gpus:
        print(f"[Warn] Ultralytics DDP requires total batch >= num_gpus. Got batch={total_batch}, gpus={n_gpus}. Auto-setting batch={n_gpus}.")
        total_batch = n_gpus

    # Ultralytics uses `workers` per rank. With multi-GPU, total workers ~= workers * num_gpus.
    # For tiny datasets (80 images), too many workers can cause long stalls/overhead.
    workers = int(args.workers)
    if n_gpus >= 2 and workers > 2:
        print(f"[Warn] Multi-GPU with small data: workers is per-rank. Got workers={workers}, gpus={n_gpus}. Auto-setting workers=2.")
        workers = 2

    results = model.train(
        data=str(yaml_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=total_batch,
        device=device,
        workers=workers,
        seed=args.seed,
        pretrained=True,
        # Augmentations (small-data friendly)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        shear=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=10,
        # Multi-GPU is incompatible with rect=True in Ultralytics
        rect=False,
        cos_lr=True,
        lr0=args.lr0,
        lrf=0.01,
        warmup_epochs=3.0,
        optimizer="AdamW",
        weight_decay=5e-4,
        cache=args.cache,
        plots=False,
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    # best weights path
    best = Path(results.save_dir) / "weights" / "best.pt"
    return best


@torch.no_grad()
def predict_to_submission(args) -> Path:
    from ultralytics import YOLO

    paths = get_paths()
    model = YOLO(args.weights)

    test_imgs = sorted(paths.test_dir.glob("*.jpg"))
    rows = []

    vis_dir = None
    if getattr(args, "vis_dir", None):
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    empty_cnt = 0
    for img_path in test_imgs:
        img_id = int(img_path.stem)
        # Ultralytics returns boxes in original image pixel coords
        preds = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        r = preds[0]
        best_xyxy = None
        best_conf = None
        if r.boxes is None or len(r.boxes) == 0:
            empty_cnt += 1
            if args.fallback == "center":
                with Image.open(img_path) as im:
                    w, h = im.size
                x, y = w / 2.0, h / 2.0
            else:
                x, y = 0.0, 0.0
        else:
            b = r.boxes
            # pick highest confidence
            confs = b.conf.detach().cpu().numpy()
            best_i = int(np.argmax(confs))
            xyxy = b.xyxy[best_i].detach().cpu().numpy().tolist()
            best_xyxy = xyxy
            best_conf = float(confs[best_i])
            x = (xyxy[0] + xyxy[2]) / 2.0
            y = (xyxy[1] + xyxy[3]) / 2.0

        if vis_dir is not None:
            with Image.open(img_path).convert("RGB") as im:
                draw = ImageDraw.Draw(im)
                if best_xyxy is not None:
                    x1, y1, x2, y2 = best_xyxy
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                    if best_conf is not None:
                        draw.text((max(0, x1), max(0, y1)), f"conf={best_conf:.3f}", fill=(255, 0, 0))
                else:
                    draw.text((5, 5), "NO DET", fill=(255, 0, 0))

                cx, cy = float(x), float(y)
                s = 6
                draw.line((cx - s, cy, cx + s, cy), fill=(0, 255, 0), width=3)
                draw.line((cx, cy - s, cx, cy + s), fill=(0, 255, 0), width=3)
                im.save(vis_dir / f"{img_id}.jpg")

        rows.append({"ImageID": f"{img_id}_Fovea_X", "value": float(x)})
        rows.append({"ImageID": f"{img_id}_Fovea_Y", "value": float(y)})

    df = pd.DataFrame(rows)
    if paths.sample_sub.exists():
        sample = pd.read_csv(paths.sample_sub)
        df = sample[["ImageID"]].merge(df, on="ImageID", how="left")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if getattr(args, "debug", False):
        print(f"[Debug] weights={args.weights} conf={args.conf} iou={args.iou} device={args.device}")
        print(f"[Debug] empty_detections={empty_cnt}/{len(test_imgs)} fallback={args.fallback}")
    return out_csv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Task4 YOLO pipeline (VOC->YOLO, train, predict submission)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--model", type=str, default="yolo11s.pt", help="Ultralytics model (e.g., yolo11n.pt/yolo11s.pt/yolov8s.pt)")
    p_train.add_argument("--imgsz", type=int, default=1024)
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--batch", type=int, default=16, help="TOTAL batch size (will be split across GPUs in DDP)")
    p_train.add_argument("--device", type=str, default="0,1,2,3,4,5,6,7")
    p_train.add_argument("--workers", type=int, default=2, help="Dataloader workers PER GPU/rank. Total ~ workers * num_gpus")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--val-ratio", type=float, default=0.2)
    p_train.add_argument("--out-root", type=str, default=str((Path(__file__).resolve().parent / "yolo_dataset").as_posix()))
    p_train.add_argument("--clean", action="store_true")
    p_train.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True, help="Cache images for faster training (recommended for 80 images)")
    p_train.add_argument("--project", type=str, default=str((Path(__file__).resolve().parent / "runs").as_posix()))
    p_train.add_argument("--name", type=str, default="task4_yolo")
    p_train.add_argument("--mosaic", type=float, default=1.0)
    p_train.add_argument("--mixup", type=float, default=0.1)
    p_train.add_argument("--lr0", type=float, default=1e-3)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--weights", type=str, required=True)
    p_pred.add_argument("--imgsz", type=int, default=1024)
    p_pred.add_argument("--conf", type=float, default=0.001)
    p_pred.add_argument("--iou", type=float, default=0.5)
    p_pred.add_argument("--device", type=str, default="0")
    p_pred.add_argument("--fallback", type=str, default="zero", choices=["zero", "center"], help="If no boxes predicted: output (0,0) or image center")
    p_pred.add_argument("--debug", action="store_true")
    p_pred.add_argument("--vis-dir", type=str, default="", help="If set, save annotated test images (bbox + predicted center) into this folder")
    p_pred.add_argument("--out-csv", type=str, default=str((Path(__file__).resolve().parent / "submission_task4.csv").as_posix()))

    return p


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(getattr(args, "seed", 42))

    if args.cmd == "train":
        best = train_yolo(args)
        print(f"Best weights: {best}")
    elif args.cmd == "predict":
        out = predict_to_submission(args)
        print(f"Wrote submission: {out}")


if __name__ == "__main__":
    main()
