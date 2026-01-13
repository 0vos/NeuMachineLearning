import argparse
import os
import random
import socket
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.models.detection import (
	FasterRCNN_ResNet50_FPN_Weights,
	fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw


@dataclass
class Paths:
	root: Path
	train_dir: Path
	test_dir: Path
	train_xml_dir: Path
	sample_sub: Path


def get_paths() -> Paths:
	root = Path(__file__).resolve().parent
	det_root = root / "detection"
	return Paths(
		root=root,
		train_dir=det_root / "train",
		test_dir=det_root / "test",
		train_xml_dir=det_root / "train_location",
		sample_sub=det_root / "sample_submission.csv",
	)


def seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def is_dist_avail_and_initialized() -> bool:
	return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
	return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
	return get_rank() == 0


def init_distributed_mode() -> tuple[bool, int, int]:
	# If already initialized (e.g., user code called init_process_group), be idempotent.
	if dist.is_available() and dist.is_initialized():
		local_rank = int(os.environ.get("LOCAL_RANK", str(torch.cuda.current_device() if torch.cuda.is_available() else 0)))
		world_size = dist.get_world_size()
		try:
			torch.cuda.set_device(local_rank)
		except Exception:
			pass
		return True, local_rank, world_size

	if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
		rank = int(os.environ["RANK"])
		world_size = int(os.environ["WORLD_SIZE"])
		local_rank = int(os.environ.get("LOCAL_RANK", "0"))
		dist.init_process_group(backend="nccl", init_method="env://")
		torch.cuda.set_device(local_rank)
		try:
			dist.barrier(device_ids=[local_rank])
		except TypeError:
			dist.barrier()
		return True, local_rank, world_size
	return False, 0, 1


def cleanup_distributed_mode() -> None:
	if is_dist_avail_and_initialized():
		try:
			dist.barrier(device_ids=[torch.cuda.current_device()])
		except Exception:
			dist.barrier()
		dist.destroy_process_group()


def _find_free_port() -> int:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind(("127.0.0.1", 0))
		return int(s.getsockname()[1])


def _ddp_spawn_worker(local_rank: int, world_size: int, master_port: int, args_dict: dict) -> None:
	os.environ["MASTER_ADDR"] = "127.0.0.1"
	os.environ["MASTER_PORT"] = str(master_port)
	os.environ["WORLD_SIZE"] = str(world_size)
	os.environ["RANK"] = str(local_rank)
	os.environ["LOCAL_RANK"] = str(local_rank)

	# Ensure each worker uses its assigned GPU.
	if torch.cuda.is_available():
		torch.cuda.set_device(local_rank)

	# Prevent recursive spawning inside workers.
	args_dict = dict(args_dict)
	args_dict["launcher"] = "none"

	worker_args = argparse.Namespace(**args_dict)
	try:
		train(worker_args)
	finally:
		# train() will cleanup; this is just a safe guard.
		cleanup_distributed_mode()



def parse_voc_xml(xml_path: Path) -> tuple[list[list[float]], tuple[int, int] | None]:
	tree = ET.parse(xml_path)
	root = tree.getroot()
	xml_size = root.find("size")
	if xml_size is not None:
		try:
			xml_w = int(xml_size.findtext("width"))
			xml_h = int(xml_size.findtext("height"))
			xml_hw: tuple[int, int] | None = (xml_w, xml_h)
		except Exception:
			xml_hw = None
	else:
		xml_hw = None

	boxes: list[list[float]] = []
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
	return boxes, xml_hw


class FoveaDetDataset(Dataset):
	def __init__(self, img_dir: Path, xml_dir: Path, ids: list[int]):
		self.img_dir = img_dir
		self.xml_dir = xml_dir
		self.ids = ids

	def __len__(self) -> int:
		return len(self.ids)

	def __getitem__(self, idx: int):
		image_id = self.ids[idx]
		img_path = self.img_dir / f"{image_id:04d}.jpg"
		xml_path = self.xml_dir / f"{image_id:04d}.xml"

		image = Image.open(img_path).convert("RGB")
		img_w, img_h = image.size
		img_tensor = F.to_tensor(image)

		boxes_list, xml_hw = parse_voc_xml(xml_path)
		if xml_hw is not None:
			xml_w, xml_h = xml_hw
			# Some samples have XML size different from actual JPG size.
			# Scale annotations to match actual image pixel coordinates.
			if xml_w > 0 and xml_h > 0 and (xml_w != img_w or xml_h != img_h):
				sx = img_w / float(xml_w)
				sy = img_h / float(xml_h)
				scaled: list[list[float]] = []
				for xmin, ymin, xmax, ymax in boxes_list:
					xmin2 = xmin * sx
					xmax2 = xmax * sx
					ymin2 = ymin * sy
					ymax2 = ymax * sy
					scaled.append([xmin2, ymin2, xmax2, ymax2])
				boxes_list = scaled
		if len(boxes_list) == 0:
			boxes = torch.zeros((0, 4), dtype=torch.float32)
			labels = torch.zeros((0,), dtype=torch.int64)
		else:
			boxes = torch.tensor(boxes_list, dtype=torch.float32)
			# Clamp to valid image bounds
			boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w - 1)
			boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h - 1)
			labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

		area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
		target = {
			"boxes": boxes,
			"labels": labels,
			"image_id": torch.tensor([image_id], dtype=torch.int64),
			"area": area,
			"iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
		}

		return img_tensor, target


class FoveaTestDataset(Dataset):
	def __init__(self, img_dir: Path):
		self.img_dir = img_dir
		self.img_names = sorted([p.name for p in img_dir.glob("*.jpg")])

	def __len__(self) -> int:
		return len(self.img_names)

	def __getitem__(self, idx: int):
		name = self.img_names[idx]
		path = self.img_dir / name
		image = Image.open(path).convert("RGB")
		return F.to_tensor(image), name


def collate_fn(batch):
	return tuple(zip(*batch))


def build_model(num_classes: int = 2):
	weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
	model = fasterrcnn_resnet50_fpn(weights=weights)

	in_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	return model


@torch.no_grad()
def evaluate_loss(model, data_loader, device, amp: bool) -> float:
    model.train()
    total = 0.0
    n = 0
    scaler_ctx = torch.cuda.amp.autocast(enabled=amp)
    for images, targets in data_loader:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        with scaler_ctx:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        total += loss.item() * len(images)
        n += len(images)
    return total / max(1, n)


def train(args) -> Path:
    paths = get_paths()
    distributed, local_rank, world_size = init_distributed_mode()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed + get_rank())

    ids = list(range(1, 81))
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    val_size = max(1, int(len(ids) * args.val_ratio))
    val_ids = sorted(ids[:val_size])
    train_ids = sorted(ids[val_size:])

    train_ds = FoveaDetDataset(paths.train_dir, paths.train_xml_dir, train_ids)
    val_ds = FoveaDetDataset(paths.train_dir, paths.train_xml_dir, val_ids)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_model(num_classes=2)
    model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pth"
    last_path = out_dir / "last.pth"

    best_val = float("inf")
    if is_main_process():
        print(f"[Train] device={device} distributed={distributed} world_size={world_size}")
        print(f"[Train] train={len(train_ds)} val={len(val_ds)} out={out_dir}")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        n = 0
        t0 = time.time()

        for images, targets in train_loader:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * len(images)
            n += len(images)

        lr_sched.step()
        train_loss = epoch_loss / max(1, n)

        # Val loss (compute on each rank; only rank0 prints/saves)
        val_loss = evaluate_loss(model.module if distributed else model, val_loader, device, amp=args.amp)

        if distributed:
            v = torch.tensor([val_loss], device=device)
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            val_loss = (v.item() / world_size)

        if is_main_process():
            dt = time.time() - t0
            print(f"Epoch {epoch:03d}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  time={dt:.1f}s")

            ckpt = {
                "epoch": epoch,
                "model": (model.module.state_dict() if distributed else model.state_dict()),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, last_path)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, best_path)

    cleanup_distributed_mode()
    return best_path


@torch.no_grad()
def predict(args) -> Path:
	paths = get_paths()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	vis_dir = None
	if getattr(args, "vis_dir", ""):
		vis_dir = Path(args.vis_dir)
		vis_dir.mkdir(parents=True, exist_ok=True)

	model = build_model(num_classes=2)
	ckpt = torch.load(args.weights, map_location="cpu")
	state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
	model.load_state_dict(state, strict=True)
	model.to(device)
	model.eval()

	test_ds = FoveaTestDataset(paths.test_dir)
	test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

	rows: list[dict] = []
	for img, name in test_loader:
		img = img.to(device, non_blocking=True)
		preds = model([img[0]])[0]
		scores = preds.get("scores")
		boxes = preds.get("boxes")

		img_id = int(Path(name[0]).stem)
		has_box = not (scores is None or boxes is None or len(scores) == 0 or float(scores[0].item()) < args.score_thr)
		if not has_box:
			pred_x, pred_y = 0.0, 0.0
		else:
			b = boxes[0].detach().cpu().numpy().tolist()
			pred_x = (b[0] + b[2]) / 2.0
			pred_y = (b[1] + b[3]) / 2.0

		if vis_dir is not None:
			img_path = paths.test_dir / name[0]
			with Image.open(img_path).convert("RGB") as im:
				draw = ImageDraw.Draw(im)
				if has_box:
					draw.rectangle(b, outline=(255, 0, 0), width=3)
					if scores is not None and len(scores) > 0:
						draw.text((max(0, b[0]), max(0, b[1])), f"score={float(scores[0].item()):.3f}", fill=(255, 0, 0))
				else:
					draw.text((5, 5), "NO DET", fill=(255, 0, 0))

				cx, cy = float(pred_x), float(pred_y)
				s = 6
				draw.line((cx - s, cy, cx + s, cy), fill=(0, 255, 0), width=3)
				draw.line((cx, cy - s, cx, cy + s), fill=(0, 255, 0), width=3)
				im.save(vis_dir / f"{img_id}.jpg")

		rows.append({"ImageID": f"{img_id}_Fovea_X", "value": float(pred_x)})
		rows.append({"ImageID": f"{img_id}_Fovea_Y", "value": float(pred_y)})

	df = pd.DataFrame(rows)

	# Match sample order if possible
	if paths.sample_sub.exists():
		sample = pd.read_csv(paths.sample_sub)
		df = sample[["ImageID"]].merge(df, on="ImageID", how="left")

	out_path = Path(args.out_csv)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(out_path, index=False)
	return out_path


def build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Task4: Fovea localization via Faster R-CNN bbox center")

	# Make subcommands optional: default to train so `python task4.py` works.
	sub = p.add_subparsers(dest="cmd")
	p.set_defaults(cmd="train")

	# Shared training args are accepted at top-level too.
	p.add_argument("--epochs", type=int, default=30)
	p.add_argument("--batch-size", type=int, default=2)
	p.add_argument("--lr", type=float, default=2e-4)
	p.add_argument("--weight-decay", type=float, default=1e-4)
	p.add_argument("--num-workers", type=int, default=4)
	p.add_argument("--val-ratio", type=float, default=0.2)
	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--amp", action="store_true")
	p.add_argument(
		"--out-dir",
		type=str,
		default=str((Path(__file__).resolve().parent / "runs" / "task4_fasterrcnn").as_posix()),
	)

	# Launcher for multi-GPU when running with plain `python`.
	p.add_argument(
		"--launcher",
		type=str,
		default="auto",
		choices=["auto", "spawn", "none"],
		help="When cmd=train and not using torchrun, auto/spawn will launch multi-GPU DDP via multiprocessing.",
	)
	p.add_argument(
		"--gpus",
		type=int,
		default=0,
		help="Number of GPUs to use for spawn/auto. 0 means use all visible GPUs.",
	)

	# Explicit subcommands remain supported.
	sub.add_parser("train", help="Train Faster R-CNN detector")
	p_pred = sub.add_parser("predict", help="Predict on test and write submission")
	p_pred.add_argument("--weights", type=str, required=True)
	p_pred.add_argument("--score-thr", type=float, default=0.0)
	p_pred.add_argument("--num-workers", type=int, default=2)
	p_pred.add_argument("--vis-dir", type=str, default="", help="If set, save annotated test images (bbox + predicted center) into this folder")
	p_pred.add_argument(
		"--out-csv",
		type=str,
		default=str((Path(__file__).resolve().parent / "submission_task4.csv").as_posix()),
	)
	return p


def main() -> None:
	args = build_argparser().parse_args()
	if args.cmd == "train":
		# If launched by torchrun, env vars will exist and train() will use DDP.
		launched_by_torchrun = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)
		if (not launched_by_torchrun) and torch.cuda.is_available() and args.launcher in {"auto", "spawn"}:
			n_all = torch.cuda.device_count()
			n = n_all if args.gpus == 0 else min(args.gpus, n_all)
			if n >= 2:
				if is_main_process():
					print(f"[Launcher] python-mode spawn DDP on {n} GPUs (of {n_all})")
				port = _find_free_port()
				# Keep args serializable for multiprocessing
				args_dict = vars(args).copy()
				mp.spawn(_ddp_spawn_worker, args=(n, port, args_dict), nprocs=n, join=True)
				return

		best = train(args)
		if is_main_process():
			print(f"Saved best checkpoint: {best}")
	elif args.cmd == "predict":
		out = predict(args)
		print(f"Wrote submission: {out}")


if __name__ == "__main__":
	main()
