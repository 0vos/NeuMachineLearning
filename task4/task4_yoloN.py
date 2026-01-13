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

    repeat_train = int(getattr(args, "repeat_train", 1))
    repeat_train = max(1, repeat_train)

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

        def _write_one(dst_stem: str):
            img_dst = out_root / "images" / split / f"{dst_stem}.jpg"
            lbl_dst = out_root / "labels" / split / f"{dst_stem}.txt"
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

        if split == "train" and repeat_train > 1:
            for r in range(repeat_train):
                _write_one(f"{img_src.stem}_r{r}")
        else:
            _write_one(img_src.stem)

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


def _val_ids_from_seed(seed: int, val_ratio: float) -> set[int]:
    ids = list(range(1, 81))
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    val_size = max(1, int(len(ids) * float(val_ratio)))
    return set(ids[:val_size])


def _gt_center_from_xml(image_id: int) -> tuple[float, float] | None:
    paths = get_paths()
    img_path = paths.train_dir / f"{image_id:04d}.jpg"
    xml_path = paths.train_xml_dir / f"{image_id:04d}.xml"
    if not img_path.exists() or not xml_path.exists():
        return None
    with Image.open(img_path) as im:
        img_w, img_h = im.size
    boxes, xml_hw = _parse_voc_xml(xml_path)
    boxes = _scale_boxes_to_image(boxes, xml_hw, (img_w, img_h))
    if not boxes:
        return None
    xmin, ymin, xmax, ymax = boxes[0]
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    return float(cx), float(cy)


@torch.no_grad()
def eval_yolo_on_val(args) -> float:
    """Evaluate a trained YOLO model on the fixed val split and return coordinate MSE.

    Metric: mean over images of ((dx^2 + dy^2) / 2).
    """
    from ultralytics import YOLO

    paths = get_paths()
    val_ids = _val_ids_from_seed(getattr(args, "seed", 42), getattr(args, "val_ratio", 0.2))

    model = YOLO(args.weights)
    fallback = str(getattr(args, "fallback", "center"))

    rows = []
    for image_id in sorted(val_ids):
        img_path = paths.train_dir / f"{image_id:04d}.jpg"
        if not img_path.exists():
            continue

        gt = _gt_center_from_xml(image_id)
        if gt is None:
            continue
        gt_x, gt_y = gt

        preds = model.predict(
            source=str(img_path),
            imgsz=int(getattr(args, "imgsz", 1024)),
            conf=float(getattr(args, "conf", 0.001)),
            iou=float(getattr(args, "iou", 0.5)),
            device=str(getattr(args, "device", "0")),
            verbose=False,
        )
        r = preds[0]
        if r.boxes is None or len(r.boxes) == 0:
            with Image.open(img_path) as im:
                w, h = im.size
            if fallback == "center":
                pred_x, pred_y = w / 2.0, h / 2.0
            else:
                pred_x, pred_y = 0.0, 0.0
            best_conf = 0.0
            status = "no_det"
        else:
            b = r.boxes
            confs = b.conf.detach().cpu().numpy()
            xyxys = b.xyxy.detach().cpu().numpy()
            best_i = int(np.argmax(confs))
            x1, y1, x2, y2 = xyxys[best_i].tolist()
            pred_x = (x1 + x2) / 2.0
            pred_y = (y1 + y2) / 2.0
            best_conf = float(confs[best_i])
            status = "ok"

        dx = float(pred_x - gt_x)
        dy = float(pred_y - gt_y)
        mse_xy = (dx * dx + dy * dy) / 2.0
        rows.append(
            {
                "id": int(image_id),
                "gt_x": float(gt_x),
                "gt_y": float(gt_y),
                "pred_x": float(pred_x),
                "pred_y": float(pred_y),
                "dx": float(dx),
                "dy": float(dy),
                "mse_xy": float(mse_xy),
                "conf": float(best_conf),
                "status": status,
            }
        )

    if not rows:
        raise RuntimeError("No validation samples found (check paths / val split / annotations).")

    df = pd.DataFrame(rows)
    mse = float(df["mse_xy"].mean())
    out_report = str(getattr(args, "out_report", "")).strip()
    if out_report:
        outp = Path(out_report)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values("mse_xy", ascending=False).to_csv(outp, index=False)
    if getattr(args, "debug", False):
        worst = df.sort_values("mse_xy", ascending=False).head(10)
        print("[Eval] worst-10 ids:", worst["id"].tolist())
    return mse


def _resolve_ensemble_weights(args) -> list[Path]:
    weights = []

    # Explicit list (space-separated)
    explicit = getattr(args, "ensemble_weights", None)
    if explicit:
        for w in explicit:
            wp = Path(str(w))
            if wp.exists():
                weights.append(wp)
            else:
                print(f"[Warn] ensemble weight not found: {wp}")

    # Pattern by index range: root/prefix{idx:04d}/weights/best.pt
    root = str(getattr(args, "ensemble_root", "")).strip()
    prefix = str(getattr(args, "ensemble_prefix", "")).strip()
    if root and prefix:
        start = int(getattr(args, "ensemble_start", 0))
        end = int(getattr(args, "ensemble_end", -1))
        suffix = str(getattr(args, "ensemble_suffix", "weights/best.pt")).strip() or "weights/best.pt"
        if end < start:
            raise ValueError(f"ensemble_end ({end}) must be >= ensemble_start ({start})")
        rootp = Path(root)
        for i in range(start, end + 1):
            wp = rootp / f"{prefix}{i:04d}" / suffix
            if wp.exists():
                weights.append(wp)
            else:
                print(f"[Warn] ensemble weight missing: {wp}")

    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for w in weights:
        ws = str(w.resolve())
        if ws in seen:
            continue
        seen.add(ws)
        uniq.append(w)
    if not uniq:
        raise RuntimeError("No ensemble weights resolved. Provide --ensemble-weights or --ensemble-root/--ensemble-prefix.")
    return uniq


@torch.no_grad()
def ensemble_predict_to_submission(args) -> Path:
    """Ensemble multiple YOLO checkpoints by aggregating predicted centers."""
    from ultralytics import YOLO

    paths = get_paths()
    weights_list = _resolve_ensemble_weights(args)
    test_imgs = sorted(paths.test_dir.glob("*.jpg"))
    if not test_imgs:
        raise RuntimeError(f"No test images found in: {paths.test_dir}")

    imgsz = int(getattr(args, "imgsz", 1024))
    conf_thr = float(getattr(args, "conf", 0.001))
    iou_thr = float(getattr(args, "iou", 0.5))
    device = str(getattr(args, "device", "0"))
    fallback = str(getattr(args, "fallback", "center"))

    mode = str(getattr(args, "ensemble_mode", "median")).lower()
    min_conf = float(getattr(args, "ensemble_min_conf", 0.0))

    n_models = len(weights_list)
    n_imgs = len(test_imgs)
    xs = np.zeros((n_models, n_imgs), dtype=np.float32)
    ys = np.zeros((n_models, n_imgs), dtype=np.float32)
    cs = np.zeros((n_models, n_imgs), dtype=np.float32)
    valid = np.zeros((n_models, n_imgs), dtype=np.bool_)

    # Preload image sizes for fallback / normalization
    img_sizes = []
    for p in test_imgs:
        with Image.open(p) as im:
            img_sizes.append(im.size)

    for mi, w in enumerate(weights_list):
        print(f"[Ensemble] ({mi+1}/{n_models}) loading: {w}")
        model = YOLO(str(w))
        for ii, img_path in enumerate(test_imgs):
            preds = model.predict(
                source=str(img_path),
                imgsz=imgsz,
                conf=conf_thr,
                iou=iou_thr,
                device=device,
                verbose=False,
            )
            r = preds[0]
            if r.boxes is None or len(r.boxes) == 0:
                w0, h0 = img_sizes[ii]
                if fallback == "center":
                    x, y = w0 / 2.0, h0 / 2.0
                else:
                    x, y = 0.0, 0.0
                c = 0.0
                ok = False
            else:
                b = r.boxes
                confs = b.conf.detach().cpu().numpy()
                xyxys = b.xyxy.detach().cpu().numpy()
                best_i = int(np.argmax(confs))
                x1, y1, x2, y2 = xyxys[best_i].tolist()
                x = (x1 + x2) / 2.0
                y = (y1 + y2) / 2.0
                c = float(confs[best_i])
                ok = c >= min_conf

            xs[mi, ii] = float(x)
            ys[mi, ii] = float(y)
            cs[mi, ii] = float(c)
            valid[mi, ii] = bool(ok)

        # Free memory between models (important for GPU).
        del model
        if torch.cuda.is_available() and device not in {"cpu", "-1"}:
            torch.cuda.empty_cache()

    # Aggregate
    agg_x = np.zeros((n_imgs,), dtype=np.float32)
    agg_y = np.zeros((n_imgs,), dtype=np.float32)
    used_counts = np.zeros((n_imgs,), dtype=np.int32)
    for ii in range(n_imgs):
        msk = valid[:, ii]
        if not np.any(msk):
            # all invalid: fallback to simple median over all (still stable) or center/zero
            if mode in {"median", "mean", "weighted"}:
                agg_x[ii] = float(np.median(xs[:, ii]))
                agg_y[ii] = float(np.median(ys[:, ii]))
            else:
                w0, h0 = img_sizes[ii]
                agg_x[ii] = float(w0 / 2.0)
                agg_y[ii] = float(h0 / 2.0)
            used_counts[ii] = 0
            continue

        xv = xs[msk, ii]
        yv = ys[msk, ii]
        cv = cs[msk, ii]
        used_counts[ii] = int(np.sum(msk))

        if mode == "mean":
            agg_x[ii] = float(np.mean(xv))
            agg_y[ii] = float(np.mean(yv))
        elif mode == "weighted":
            ww = np.maximum(cv, 1e-6)
            ww = ww / float(np.sum(ww))
            agg_x[ii] = float(np.sum(xv * ww))
            agg_y[ii] = float(np.sum(yv * ww))
        else:
            # default median (robust)
            agg_x[ii] = float(np.median(xv))
            agg_y[ii] = float(np.median(yv))

    rows = []
    for ii, img_path in enumerate(test_imgs):
        img_id = int(img_path.stem)
        rows.append({"ImageID": f"{img_id}_Fovea_X", "value": float(agg_x[ii])})
        rows.append({"ImageID": f"{img_id}_Fovea_Y", "value": float(agg_y[ii])})

    df = pd.DataFrame(rows)
    if paths.sample_sub.exists():
        sample = pd.read_csv(paths.sample_sub)
        df = sample[["ImageID"]].merge(df, on="ImageID", how="left")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if getattr(args, "debug", False):
        print(f"[Ensemble] mode={mode} n_models={n_models} min_conf={min_conf}")
        print(f"[Ensemble] used_models_per_image: min={int(used_counts.min())} mean={float(used_counts.mean()):.2f} max={int(used_counts.max())}")
        print(f"[Ensemble] wrote: {out_csv}")
    return out_csv


def tune_with_optuna(args) -> dict:
    """Run Optuna HPO over YOLO training hyperparameters, optimizing val MSE."""
    try:
        import optuna
    except Exception as e:
        raise RuntimeError(
            "Optuna is not installed. Install it in your env, e.g. `pip install optuna`."
        ) from e

    import json

    base_name = str(getattr(args, "base_name", "task4_optuna"))
    study_name = str(getattr(args, "study_name", "task4_yolo_tune"))
    storage = str(getattr(args, "storage", "")).strip() or None
    sampler_name = str(getattr(args, "sampler", "tpe")).lower()

    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=int(getattr(args, "seed", 42)))
    else:
        sampler = optuna.samplers.TPESampler(seed=int(getattr(args, "seed", 42)))

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )

    fixed_seed = int(getattr(args, "seed", 42))
    fixed_val_ratio = float(getattr(args, "val_ratio", 0.2))

    def _objective(trial: "optuna.trial.Trial") -> float:
        # Copy args into per-trial namespace.
        targs = argparse.Namespace(**vars(args))

        # Sample search space (keep it small + stable for tiny dataset).
        preset = str(getattr(args, "preset", "strong"))
        targs.preset = preset

        targs.imgsz = trial.suggest_categorical("imgsz", [768, 896, 1024, 1280])
        targs.lr0 = trial.suggest_float("lr0", 1e-5, 2e-3, log=True)
        targs.lrf = trial.suggest_float("lrf", 0.005, 0.05, log=True)
        targs.weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
        targs.warmup_epochs = trial.suggest_float("warmup_epochs", 1.0, 8.0)
        targs.close_mosaic = trial.suggest_int("close_mosaic", 0, 30)
        targs.degrees = trial.suggest_float("degrees", 0.0, 6.0)
        targs.translate = trial.suggest_float("translate", 0.0, 0.15)
        targs.scale = trial.suggest_float("scale", 0.05, 0.45)
        targs.mosaic = trial.suggest_float("mosaic", 0.0, 0.8)
        targs.mixup = trial.suggest_float("mixup", 0.0, 0.2)

        # Keep split identical across trials.
        targs.seed = fixed_seed
        targs.val_ratio = fixed_val_ratio

        # Per-trial naming
        targs.name = f"{base_name}_t{trial.number:04d}"

        # Train
        best = train_yolo(targs)

        # Eval on val with a fixed inference setup.
        eval_imgsz = int(getattr(args, "eval_imgsz", 0))
        eval_imgsz = int(targs.imgsz) if eval_imgsz <= 0 else eval_imgsz
        eargs = argparse.Namespace(
            weights=str(best),
            imgsz=eval_imgsz,
            conf=float(getattr(args, "eval_conf", 0.001)),
            iou=float(getattr(args, "eval_iou", 0.5)),
            device=str(getattr(args, "device", "0")),
            seed=fixed_seed,
            val_ratio=fixed_val_ratio,
            fallback=str(getattr(args, "eval_fallback", "center")),
            out_report="",
            debug=False,
        )
        mse = eval_yolo_on_val(eargs)
        trial.set_user_attr("best_weights", str(best))
        return float(mse)

    n_trials = int(getattr(args, "trials", 20))
    timeout = float(getattr(args, "timeout", 0.0))
    if n_trials > 0:
        study.optimize(
            _objective,
            n_trials=n_trials,
            timeout=(None if timeout <= 0 else timeout),
            gc_after_trial=True,
            catch=(Exception,),
        )
    else:
        # Reuse existing study results (useful if trials already ran but final training failed).
        if len(study.trials) == 0:
            raise RuntimeError("No existing trials found in this study; cannot reuse best_params. Run with --trials > 0 first.")
        if study.best_trial is None:
            raise RuntimeError("Study has no successful trials; cannot reuse best_params.")
        print(f"[Tune] --trials={n_trials}. Reusing existing study best_params (no new trials).")

    best = {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "best_trial": int(study.best_trial.number),
        "best_weights": str(study.best_trial.user_attrs.get("best_weights", "")),
        "study_name": study.study_name,
        "storage": storage or "in_memory",
    }

    # Optional: automatically run a final training using best params (no manual copy/paste).
    final_model = str(getattr(args, "final_model", "")).strip()
    if final_model:
        fargs = argparse.Namespace(**vars(args))
        fargs.model = final_model
        fargs.name = str(getattr(args, "final_name", "task4_yolo_optuna_final"))
        fargs.project = str(getattr(args, "final_project", "")).strip() or str(getattr(args, "project", ""))
        fargs.device = str(getattr(args, "final_device", "")).strip() or str(getattr(args, "device", "0"))

        final_epochs = int(getattr(args, "final_epochs", 0))
        if final_epochs > 0:
            fargs.epochs = final_epochs
        final_batch = int(getattr(args, "final_batch", 0))
        if final_batch > 0:
            fargs.batch = final_batch

        final_repeat = int(getattr(args, "final_repeat_train", 0))
        if final_repeat > 0:
            fargs.repeat_train = final_repeat

        final_out_root = str(getattr(args, "final_out_root", "")).strip()
        if final_out_root:
            fargs.out_root = final_out_root
        fargs.clean = bool(getattr(args, "final_clean", False))

        # Apply best params onto final args.
        bp = best["best_params"]
        for k in [
            "imgsz",
            "lr0",
            "lrf",
            "weight_decay",
            "warmup_epochs",
            "close_mosaic",
            "degrees",
            "translate",
            "scale",
            "mosaic",
            "mixup",
        ]:
            if k in bp:
                setattr(fargs, k, bp[k])

        # Allow explicit override for final imgsz (common when tuning with smaller model).
        final_imgsz = int(getattr(args, "final_imgsz", 0))
        if final_imgsz > 0:
            fargs.imgsz = final_imgsz

        print("[Tune] launching final train with best_params + final overrides...")
        try:
            final_best = train_yolo(fargs)
            best["final_weights"] = str(final_best)
            best["final_name"] = str(fargs.name)
        except torch.OutOfMemoryError as e:
            # Don't lose the tuning results just because final training OOM'ed.
            best["final_error"] = "cuda_oom"
            best["final_error_detail"] = str(e)
            print("[Tune][Error] Final training failed due to CUDA OOM. Tuning results are kept; you can rerun final only.")
            print("[Tune][Hint] Try a smaller --final-batch (e.g. 16/32) or smaller --final-imgsz (e.g. 1024), or switch to a free GPU.")
            print("[Tune][Hint] To rerun final without rerunning trials: keep same --storage/--study-name and set --trials 0.")
    out_json = str(getattr(args, "out_json", "")).strip()
    if out_json:
        outp = Path(out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    return best


def train_yolo(args) -> Path:
    from ultralytics import YOLO

    yaml_path = prepare_yolo_dataset(args)

    # Resolve model name/aliases.
    model_name = str(args.model)
    if model_name.lower().startswith("yolo18"):
        # Ultralytics does not provide a standard "yolo18" family.
        # Map user intent ("stronger model") to a known strong checkpoint name.
        print(f"[Warn] '{model_name}' is not a standard Ultralytics model name. Using 'yolo11x.pt' instead.")
        model_name = "yolo11x.pt"

    # Presets override some hyperparameters (unless user explicitly sets them).
    preset = str(getattr(args, "preset", "baseline"))
    imgsz = int(args.imgsz)
    epochs = int(args.epochs)
    total_batch = int(args.batch)
    lr0 = float(args.lr0)
    lrf = float(getattr(args, "lrf", 0.01))
    warmup_epochs = float(getattr(args, "warmup_epochs", 3.0))
    optimizer = str(getattr(args, "optimizer", "AdamW"))
    weight_decay = float(getattr(args, "weight_decay", 5e-4))
    cos_lr = bool(getattr(args, "cos_lr", True))
    close_mosaic = int(getattr(args, "close_mosaic", 10))
    degrees = float(getattr(args, "degrees", 5.0))
    translate = float(getattr(args, "translate", 0.1))
    scale = float(getattr(args, "scale", 0.4))
    shear = float(getattr(args, "shear", 0.0))
    fliplr = float(getattr(args, "fliplr", 0.5))
    flipud = float(getattr(args, "flipud", 0.0))
    mosaic = float(args.mosaic)
    mixup = float(args.mixup)
    patience = int(getattr(args, "patience", 50))

    if preset == "strong":
        # Stronger defaults for precision on this task.
        # - Higher resolution
        # - Larger model
        # - Less aggressive mixup (can hurt geometric precision)
        # - Longer training with cosine LR
        if args.model == build_parser().get_default("model"):
            model_name = "yolo11x.pt"
        if args.imgsz == build_parser().get_default("imgsz"):
            imgsz = 1280
        if args.epochs == build_parser().get_default("epochs"):
            epochs = 400
        if args.batch == build_parser().get_default("batch"):
            total_batch = 64
        if args.lr0 == build_parser().get_default("lr0"):
            lr0 = 5e-4
        if getattr(args, "lrf", build_parser().get_default("lrf")) == build_parser().get_default("lrf"):
            lrf = 0.01
        if getattr(args, "warmup_epochs", build_parser().get_default("warmup_epochs")) == build_parser().get_default("warmup_epochs"):
            warmup_epochs = 5.0
        if getattr(args, "weight_decay", build_parser().get_default("weight_decay")) == build_parser().get_default("weight_decay"):
            weight_decay = 1e-4
        if getattr(args, "optimizer", build_parser().get_default("optimizer")) == build_parser().get_default("optimizer"):
            optimizer = "AdamW"
        if args.mixup == build_parser().get_default("mixup"):
            mixup = 0.0
        if args.mosaic == build_parser().get_default("mosaic"):
            mosaic = 0.5
        if getattr(args, "close_mosaic", build_parser().get_default("close_mosaic")) == build_parser().get_default("close_mosaic"):
            close_mosaic = 20
        if getattr(args, "degrees", build_parser().get_default("degrees")) == build_parser().get_default("degrees"):
            degrees = 2.0
        if getattr(args, "translate", build_parser().get_default("translate")) == build_parser().get_default("translate"):
            translate = 0.06
        if getattr(args, "scale", build_parser().get_default("scale")) == build_parser().get_default("scale"):
            scale = 0.25
        if getattr(args, "patience", build_parser().get_default("patience")) == build_parser().get_default("patience"):
            patience = 80

    print(
        "[Train] preset=%s model=%s imgsz=%s epochs=%s batch=%s lr0=%g optimizer=%s wd=%g mosaic=%g mixup=%g"
        % (preset, model_name, imgsz, epochs, total_batch, lr0, optimizer, weight_decay, mosaic, mixup)
    )

    model = YOLO(model_name)

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
        imgsz=imgsz,
        epochs=epochs,
        batch=total_batch,
        device=device,
        workers=workers,
        seed=args.seed,
        pretrained=True,
        # Augmentations (small-data friendly)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        fliplr=fliplr,
        flipud=flipud,
        mosaic=mosaic,
        mixup=mixup,
        close_mosaic=close_mosaic,
        # Multi-GPU is incompatible with rect=True in Ultralytics
        rect=False,
        cos_lr=cos_lr,
        lr0=lr0,
        lrf=lrf,
        warmup_epochs=warmup_epochs,
        optimizer=optimizer,
        weight_decay=weight_decay,
        patience=patience,
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

    aux_model = None
    aux_weights = str(getattr(args, "aux_weights", "")).strip()
    if aux_weights:
        aux_path = Path(aux_weights)
        if aux_path.exists():
            aux_model = YOLO(str(aux_path))
        else:
            print(f"[Warn] aux-weights not found: {aux_path}. Ignoring aux model.")
            aux_model = None

    test_imgs = sorted(paths.test_dir.glob("*.jpg"))
    rows = []

    vis_dir = None
    if getattr(args, "vis_dir", None):
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    empty_cnt = 0
    lowconf_cnt = 0
    aux_used_cnt = 0
    match_used_cnt = 0
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
        primary_xyxy = None
        primary_conf = None
        aux_xyxy = None
        aux_conf = None
        status = "ok"
        if r.boxes is None or len(r.boxes) == 0:
            empty_cnt += 1
            if args.fallback == "center":
                with Image.open(img_path) as im:
                    w, h = im.size
                x, y = w / 2.0, h / 2.0
            else:
                x, y = 0.0, 0.0
            status = "no_det"
        else:
            b = r.boxes
            # pick highest confidence
            confs = b.conf.detach().cpu().numpy()
            xyxys = b.xyxy.detach().cpu().numpy()
            best_i = int(np.argmax(confs))
            xyxy = xyxys[best_i].tolist()
            primary_xyxy = xyxy
            primary_conf = float(confs[best_i])

            best_xyxy = xyxy
            best_conf = float(confs[best_i])
            x = (xyxy[0] + xyxy[2]) / 2.0
            y = (xyxy[1] + xyxy[3]) / 2.0

            # If primary model is low-confidence, optionally use aux model's prediction.
            aux_switch_conf = float(getattr(args, "aux_switch_conf", 0.0))
            if aux_model is not None and aux_switch_conf > 0.0 and best_conf < aux_switch_conf:
                aux_imgsz = int(getattr(args, "aux_imgsz", 0))
                aux_imgsz = int(args.imgsz) if aux_imgsz <= 0 else aux_imgsz
                aux_conf_thr = float(getattr(args, "aux_conf", -1.0))
                aux_conf_thr = float(args.conf) if aux_conf_thr < 0 else aux_conf_thr
                aux_iou_thr = float(getattr(args, "aux_iou", -1.0))
                aux_iou_thr = float(args.iou) if aux_iou_thr < 0 else aux_iou_thr
                aux_dev = str(getattr(args, "aux_device", "")).strip() or str(args.device)
                preds_aux = aux_model.predict(
                    source=str(img_path),
                    imgsz=aux_imgsz,
                    conf=aux_conf_thr,
                    iou=aux_iou_thr,
                    device=aux_dev,
                    verbose=False,
                )
                ra = preds_aux[0]
                if ra.boxes is not None and len(ra.boxes) > 0:
                    ba = ra.boxes
                    confsa = ba.conf.detach().cpu().numpy()
                    xyxysa = ba.xyxy.detach().cpu().numpy()
                    ia = int(np.argmax(confsa))
                    aux_xyxy = xyxysa[ia].tolist()
                    aux_conf = float(confsa[ia])
                    x = (aux_xyxy[0] + aux_xyxy[2]) / 2.0
                    y = (aux_xyxy[1] + aux_xyxy[3]) / 2.0
                    best_xyxy = aux_xyxy
                    best_conf = aux_conf
                    aux_used_cnt += 1
                    status = "aux_used"

            # New: aux-guided selection among top-K primary boxes.
            # Idea: keep multiple primary candidates; if aux exists, pick the primary box whose center is closest
            # to aux center (or use a hybrid score). This is useful when primary has many low-conf boxes.
            match_topk = int(getattr(args, "match_aux_topk", 0))
            if aux_model is not None and match_topk > 0:
                # We need an aux reference center. Use aux best box center (even if we didn't switch).
                if aux_xyxy is None:
                    aux_imgsz = int(getattr(args, "aux_imgsz", 0))
                    aux_imgsz = int(args.imgsz) if aux_imgsz <= 0 else aux_imgsz
                    aux_conf_thr = float(getattr(args, "aux_conf", -1.0))
                    aux_conf_thr = float(args.conf) if aux_conf_thr < 0 else aux_conf_thr
                    aux_iou_thr = float(getattr(args, "aux_iou", -1.0))
                    aux_iou_thr = float(args.iou) if aux_iou_thr < 0 else aux_iou_thr
                    aux_dev = str(getattr(args, "aux_device", "")).strip() or str(args.device)
                    preds_aux = aux_model.predict(
                        source=str(img_path),
                        imgsz=aux_imgsz,
                        conf=aux_conf_thr,
                        iou=aux_iou_thr,
                        device=aux_dev,
                        verbose=False,
                    )
                    ra = preds_aux[0]
                    if ra.boxes is not None and len(ra.boxes) > 0:
                        ba = ra.boxes
                        confsa = ba.conf.detach().cpu().numpy()
                        xyxysa = ba.xyxy.detach().cpu().numpy()
                        ia = int(np.argmax(confsa))
                        aux_xyxy = xyxysa[ia].tolist()
                        aux_conf = float(confsa[ia])

                if aux_xyxy is not None:
                    ax1, ay1, ax2, ay2 = aux_xyxy
                    aux_cx = (ax1 + ax2) / 2.0
                    aux_cy = (ay1 + ay2) / 2.0

                    k = max(1, min(match_topk, len(confs)))
                    idx = np.argsort(-confs)[:k]
                    # distance normalization uses image diagonal
                    with Image.open(img_path) as im:
                        w, h = im.size
                    diag = float((w * w + h * h) ** 0.5) + 1e-9
                    beta = float(getattr(args, "match_aux_beta", 0.0))
                    max_norm_dist = float(getattr(args, "match_aux_max_dist", 1.0))

                    best_j = None
                    best_score = None
                    best_dist = None
                    for j in idx:
                        x1, y1, x2, y2 = xyxys[j].tolist()
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        d = float(((cx - aux_cx) ** 2 + (cy - aux_cy) ** 2) ** 0.5)
                        dn = d / diag
                        if dn > max_norm_dist:
                            continue
                        confj = float(confs[j])
                        if beta > 0.0:
                            score = confj - beta * dn
                        else:
                            score = -dn  # pure closest
                        if best_score is None or score > best_score:
                            best_score = score
                            best_j = int(j)
                            best_dist = dn

                    if best_j is not None:
                        # only override if it's different from argmax
                        if best_j != best_i:
                            chosen = xyxys[best_j].tolist()
                            x = (chosen[0] + chosen[2]) / 2.0
                            y = (chosen[1] + chosen[3]) / 2.0
                            best_xyxy = chosen
                            best_conf = float(confs[best_j])
                            match_used_cnt += 1
                            status = f"match_aux(d={best_dist:.3f})"

            lowconf_thr = float(getattr(args, "lowconf_thr", 0.0))
            lowconf_mode = str(getattr(args, "lowconf_mode", "center"))
            lowconf_topk = int(getattr(args, "lowconf_topk", 5))
            if lowconf_thr > 0.0 and best_conf < lowconf_thr:
                lowconf_cnt += 1
                with Image.open(img_path) as im:
                    w, h = im.size
                if lowconf_mode == "topk_mean" or lowconf_mode == "topk_median":
                    k = max(1, min(lowconf_topk, len(confs)))
                    idx = np.argsort(-confs)[:k]
                    centers = []
                    weights = []
                    for j in idx:
                        x1, y1, x2, y2 = xyxys[j].tolist()
                        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
                        weights.append(float(confs[j]))
                    if lowconf_mode == "topk_mean":
                        ww = np.array(weights, dtype=np.float32)
                        ww = ww / (ww.sum() + 1e-9)
                        cx = float(np.sum([c[0] * ww[i] for i, c in enumerate(centers)]))
                        cy = float(np.sum([c[1] * ww[i] for i, c in enumerate(centers)]))
                        x, y = cx, cy
                    else:
                        xs = np.array([c[0] for c in centers], dtype=np.float32)
                        ys = np.array([c[1] for c in centers], dtype=np.float32)
                        x, y = float(np.median(xs)), float(np.median(ys))
                    status = f"lowconf_{lowconf_mode}"
                else:
                    # default: center fallback
                    x, y = w / 2.0, h / 2.0
                    status = "lowconf_center"

        if vis_dir is not None:
            with Image.open(img_path).convert("RGB") as im:
                draw = ImageDraw.Draw(im)
                # draw primary box (red) if any
                if primary_xyxy is not None:
                    x1, y1, x2, y2 = primary_xyxy
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                    if primary_conf is not None:
                        draw.text((max(0, x1), max(0, y1)), f"p={primary_conf:.3f}", fill=(255, 0, 0))
                else:
                    draw.text((5, 5), "NO DET", fill=(255, 0, 0))

                # draw aux box (blue) if used
                if aux_xyxy is not None:
                    ax1, ay1, ax2, ay2 = aux_xyxy
                    draw.rectangle([ax1, ay1, ax2, ay2], outline=(0, 128, 255), width=3)
                    if aux_conf is not None:
                        draw.text((max(0, ax1), max(0, ay1) + 18), f"a={aux_conf:.3f}", fill=(0, 128, 255))

                if status != "ok":
                    draw.text((5, 25), f"{status}", fill=(255, 0, 0))

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
        if aux_model is not None and float(getattr(args, "aux_switch_conf", 0.0)) > 0.0:
            print(f"[Debug] aux_used={aux_used_cnt}/{len(test_imgs)} aux_switch_conf={getattr(args, 'aux_switch_conf', 0.0)} aux_weights={getattr(args, 'aux_weights', '')}")
        if aux_model is not None and int(getattr(args, "match_aux_topk", 0)) > 0:
            print(f"[Debug] match_aux_used={match_used_cnt}/{len(test_imgs)} match_aux_topk={getattr(args, 'match_aux_topk', 0)}")
        if float(getattr(args, "lowconf_thr", 0.0)) > 0.0:
            print(f"[Debug] lowconf_triggered={lowconf_cnt}/{len(test_imgs)} lowconf_thr={getattr(args, 'lowconf_thr', 0.0)} lowconf_mode={getattr(args, 'lowconf_mode', 'center')}")
    return out_csv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Task4 YOLO pipeline (VOC->YOLO, train, predict submission)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument(
        "--preset",
        type=str,
        default="baseline",
        choices=["baseline", "strong"],
        help="Training preset. 'strong' uses a larger model + higher imgsz + safer aug for precision.",
    )
    p_train.add_argument("--model", type=str, default="yolo11s.pt", help="Ultralytics model (e.g., yolo11n.pt/yolo11s.pt/yolo11x.pt/yolov8x.pt). 'yolo18*' will be mapped.")
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
    p_train.add_argument("--close-mosaic", type=int, default=10)
    p_train.add_argument("--degrees", type=float, default=5.0)
    p_train.add_argument("--translate", type=float, default=0.1)
    p_train.add_argument("--scale", type=float, default=0.4)
    p_train.add_argument("--shear", type=float, default=0.0)
    p_train.add_argument("--fliplr", type=float, default=0.5)
    p_train.add_argument("--flipud", type=float, default=0.0)
    p_train.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True)
    p_train.add_argument("--lr0", type=float, default=1e-3)
    p_train.add_argument("--lrf", type=float, default=0.01)
    p_train.add_argument("--warmup-epochs", type=float, default=3.0)
    p_train.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "SGD"])
    p_train.add_argument("--weight-decay", type=float, default=5e-4)
    p_train.add_argument("--patience", type=int, default=50, help="Early-stopping patience (Ultralytics).")
    p_train.add_argument("--repeat-train", type=int, default=1, help="Repeat/upsample training images by simple duplication (only train split).")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--weights", type=str, required=True)
    p_pred.add_argument("--imgsz", type=int, default=1024)
    p_pred.add_argument("--conf", type=float, default=0.001)
    p_pred.add_argument("--iou", type=float, default=0.5)
    p_pred.add_argument("--device", type=str, default="0")
    p_pred.add_argument("--fallback", type=str, default="zero", choices=["zero", "center"], help="If no boxes predicted: output (0,0) or image center")
    p_pred.add_argument("--aux-weights", type=str, default="", help="Optional: aux YOLO weights used when primary is low-confidence (e.g., yolo11n best.pt)")
    p_pred.add_argument("--aux-switch-conf", type=float, default=0.0, help="If primary best conf < this, try aux model (0 disables)")
    p_pred.add_argument("--aux-imgsz", type=int, default=0, help="Aux inference imgsz (0 means use --imgsz)")
    p_pred.add_argument("--aux-conf", type=float, default=-1.0, help="Aux conf threshold (-1 means use --conf)")
    p_pred.add_argument("--aux-iou", type=float, default=-1.0, help="Aux iou threshold (-1 means use --iou)")
    p_pred.add_argument("--aux-device", type=str, default="", help="Aux device (empty means use --device)")
    p_pred.add_argument("--match-aux-topk", type=int, default=0, help="If >0 and aux is provided: select among top-K primary boxes the one closest to aux center")
    p_pred.add_argument("--match-aux-beta", type=float, default=0.0, help="If >0: use hybrid score conf - beta*dist_norm; else pick purely closest")
    p_pred.add_argument("--match-aux-max-dist", type=float, default=1.0, help="Max normalized distance (by image diagonal) allowed for matching")
    p_pred.add_argument("--lowconf-thr", type=float, default=0.0, help="If best box conf < this, trigger low-confidence safety fallback (0 disables)")
    p_pred.add_argument("--lowconf-mode", type=str, default="center", choices=["center", "topk_mean", "topk_median"], help="Fallback mode when lowconf is triggered")
    p_pred.add_argument("--lowconf-topk", type=int, default=5, help="Top-k used by topk_mean/topk_median modes")
    p_pred.add_argument("--debug", action="store_true")
    p_pred.add_argument("--vis-dir", type=str, default="", help="If set, save annotated test images (bbox + predicted center) into this folder")
    p_pred.add_argument("--out-csv", type=str, default=str((Path(__file__).resolve().parent / "submission_task4.csv").as_posix()))

    p_eval = sub.add_parser("eval", help="Evaluate YOLO weights on the fixed val split (compute coordinate MSE)")
    p_eval.add_argument("--weights", type=str, required=True)
    p_eval.add_argument("--imgsz", type=int, default=1024)
    p_eval.add_argument("--conf", type=float, default=0.001)
    p_eval.add_argument("--iou", type=float, default=0.5)
    p_eval.add_argument("--device", type=str, default="0")
    p_eval.add_argument("--fallback", type=str, default="center", choices=["zero", "center"])
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--val-ratio", type=float, default=0.2)
    p_eval.add_argument("--out-report", type=str, default="", help="Optional CSV report path (sorted by mse desc)")
    p_eval.add_argument("--debug", action="store_true")

    p_ens = sub.add_parser("ensemble", help="Ensemble multiple YOLO weights and write submission")
    p_ens.add_argument("--imgsz", type=int, default=1024)
    p_ens.add_argument("--conf", type=float, default=0.001)
    p_ens.add_argument("--iou", type=float, default=0.5)
    p_ens.add_argument("--device", type=str, default="0")
    p_ens.add_argument("--fallback", type=str, default="center", choices=["zero", "center"])
    p_ens.add_argument("--out-csv", type=str, default=str((Path(__file__).resolve().parent / "submission_task4.csv").as_posix()))
    p_ens.add_argument("--debug", action="store_true")

    # Weight sources:
    p_ens.add_argument("--ensemble-weights", nargs="*", default=None, help="Explicit weight paths (space-separated)")
    p_ens.add_argument("--ensemble-root", type=str, default="", help="Root folder containing trial runs")
    p_ens.add_argument("--ensemble-prefix", type=str, default="", help="Run name prefix, e.g. task4_optuna_s_t")
    p_ens.add_argument("--ensemble-start", type=int, default=0)
    p_ens.add_argument("--ensemble-end", type=int, default=-1)
    p_ens.add_argument("--ensemble-suffix", type=str, default="weights/best.pt", help="Suffix under each run dir")

    # Aggregation:
    p_ens.add_argument("--ensemble-mode", type=str, default="median", choices=["median", "mean", "weighted"], help="How to aggregate centers")
    p_ens.add_argument("--ensemble-min-conf", type=float, default=0.0, help="Only use per-model prediction if conf >= this")

    p_tune = sub.add_parser("tune", help="Optuna HPO: sample hyperparams -> train -> eval val MSE")
    p_tune.add_argument("--preset", type=str, default="strong", choices=["baseline", "strong"])
    p_tune.add_argument("--model", type=str, default="yolo11s.pt")
    p_tune.add_argument("--epochs", type=int, default=120)
    p_tune.add_argument("--batch", type=int, default=16)
    p_tune.add_argument("--device", type=str, default="0")
    p_tune.add_argument("--workers", type=int, default=2)
    p_tune.add_argument("--seed", type=int, default=42)
    p_tune.add_argument("--val-ratio", type=float, default=0.2)
    p_tune.add_argument("--repeat-train", type=int, default=1)
    p_tune.add_argument("--out-root", type=str, default=str((Path(__file__).resolve().parent / "yolo_dataset").as_posix()))
    p_tune.add_argument("--clean", action="store_true")
    p_tune.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    p_tune.add_argument("--project", type=str, default=str((Path(__file__).resolve().parent / "runs").as_posix()))
    p_tune.add_argument("--base-name", type=str, default="task4_optuna")
    p_tune.add_argument("--trials", type=int, default=20)
    p_tune.add_argument("--timeout", type=float, default=0.0, help="Seconds (0 disables)")
    p_tune.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"])
    p_tune.add_argument("--study-name", type=str, default="task4_yolo_tune")
    p_tune.add_argument("--storage", type=str, default="", help="Optuna storage URL, e.g. sqlite:///task4_optuna.db")
    p_tune.add_argument("--out-json", type=str, default="", help="Optional: write best summary JSON")
    # Fixed eval setup during tuning
    p_tune.add_argument("--eval-conf", type=float, default=0.001)
    p_tune.add_argument("--eval-iou", type=float, default=0.5)
    p_tune.add_argument("--eval-imgsz", type=int, default=0, help="0 means use trial imgsz")
    p_tune.add_argument("--eval-fallback", type=str, default="center", choices=["zero", "center"])

    # Optional final training (runs automatically after tuning if --final-model is set)
    p_tune.add_argument("--final-model", type=str, default="", help="If set: run a final training after tuning using best_params")
    p_tune.add_argument("--final-name", type=str, default="task4_yolo_optuna_final")
    p_tune.add_argument("--final-project", type=str, default="")
    p_tune.add_argument("--final-device", type=str, default="")
    p_tune.add_argument("--final-epochs", type=int, default=0)
    p_tune.add_argument("--final-batch", type=int, default=0)
    p_tune.add_argument("--final-imgsz", type=int, default=0, help="Override imgsz for final training (0 uses best_params/imgsz)")
    p_tune.add_argument("--final-repeat-train", type=int, default=0)
    p_tune.add_argument("--final-out-root", type=str, default="")
    p_tune.add_argument("--final-clean", action="store_true")

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
    elif args.cmd == "eval":
        # align arg names for eval function
        eargs = argparse.Namespace(
            weights=str(args.weights),
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            iou=float(args.iou),
            device=str(args.device),
            seed=int(args.seed),
            val_ratio=float(getattr(args, "val_ratio", 0.2)),
            fallback=str(args.fallback),
            out_report=str(getattr(args, "out_report", "")),
            debug=bool(getattr(args, "debug", False)),
        )
        mse = eval_yolo_on_val(eargs)
        print(f"Val MSE(xy): {mse:.6f}")
        if getattr(args, "out_report", ""):
            print(f"Wrote report: {args.out_report}")
    elif args.cmd == "tune":
        best = tune_with_optuna(args)
        print("[Tune] best_value:", best["best_value"])
        print("[Tune] best_params:", best["best_params"])
        if best.get("best_weights"):
            print("[Tune] best_weights:", best["best_weights"])
        if best.get("final_weights"):
            print("[Tune] final_weights:", best["final_weights"])
    elif args.cmd == "ensemble":
        out = ensemble_predict_to_submission(args)
        print(f"Wrote ensemble submission: {out}")


if __name__ == "__main__":
    main()
