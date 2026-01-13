import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
	import random

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = False
	torch.backends.cudnn.benchmark = True


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def pil_to_np_gray(image: Image.Image) -> np.ndarray:
	return np.array(image.convert("L"))


def load_rgb(path: Path) -> np.ndarray:
	return np.array(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
	# Ground-truth label is grayscale with values {0,255} where 255 is vessel.
	arr = pil_to_np_gray(Image.open(path))
	return (arr > 0).astype(np.uint8)


def resize_mask_nearest(mask: np.ndarray, size: int) -> np.ndarray:
	if mask.shape[0] == size and mask.shape[1] == size:
		return mask
	return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)


@dataclass
class TrainConfig:
	data_root: Path
	img_size: int = 512
	batch_size: int = 4
	epochs: int = 40
	lr: float = 3e-4
	weight_decay: float = 1e-4
	val_ratio: float = 0.2
	seed: int = 42
	num_workers: int = 2
	encoder: str = "resnet34"
	encoder_weights: str = "imagenet"
	model: str = "unetplusplus"  # unet / unetplusplus
	amp: bool = True


class FundusSegDataset(Dataset):
	def __init__(self, image_paths: list[Path], mask_paths: list[Path] | None, aug):
		self.image_paths = image_paths
		self.mask_paths = mask_paths
		self.aug = aug

	def __len__(self) -> int:
		return len(self.image_paths)

	def __getitem__(self, idx: int):
		image = load_rgb(self.image_paths[idx])
		mask = None
		if self.mask_paths is not None:
			mask = load_mask(self.mask_paths[idx])

		if mask is None:
			transformed = self.aug(image=image)
			image = transformed["image"]
			return image

		transformed = self.aug(image=image, mask=mask)
		image = transformed["image"]
		mask_t = transformed["mask"]
		# Depending on Albumentations version / transform stack,
		# mask can be np.ndarray (H,W) or torch.Tensor (H,W).
		if isinstance(mask_t, np.ndarray):
			mask_t = torch.from_numpy(mask_t)
		mask_t = mask_t.float()
		if mask_t.ndim == 2:
			mask_t = mask_t.unsqueeze(0)
		return image, mask_t


def build_augs(img_size: int, train: bool):
	import albumentations as A
	from albumentations.pytorch import ToTensorV2

	if train:
		return A.Compose(
			[
				A.Resize(img_size, img_size, interpolation=1),
				A.HorizontalFlip(p=0.5),
				A.VerticalFlip(p=0.2),
				A.RandomRotate90(p=0.5),
				A.ShiftScaleRotate(
					shift_limit=0.05,
					scale_limit=0.15,
					rotate_limit=15,
					border_mode=0,
					p=0.5,
				),
				A.RandomBrightnessContrast(p=0.5),
				A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
				ToTensorV2(),
			]
		)

	return A.Compose(
		[
			A.Resize(img_size, img_size, interpolation=1),
			A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			ToTensorV2(),
		]
	)


def build_model(cfg: TrainConfig):
	import segmentation_models_pytorch as smp

	model_name = cfg.model.lower()
	if model_name in {"unet", "u-net"}:
		return smp.Unet(
			encoder_name=cfg.encoder,
			encoder_weights=cfg.encoder_weights,
			in_channels=3,
			classes=1,
		)
	if model_name in {"unetplusplus", "unet++", "unet_plus_plus"}:
		return smp.UnetPlusPlus(
			encoder_name=cfg.encoder,
			encoder_weights=cfg.encoder_weights,
			in_channels=3,
			classes=1,
		)
	raise ValueError(f"Unknown model: {cfg.model}")


def dice_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
	probs = torch.sigmoid(logits)
	preds = (probs > 0.5).float()
	targets = (targets > 0.5).float()
	intersection = (preds * targets).sum(dim=(1, 2, 3))
	union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
	return ((2 * intersection + eps) / (union + eps)).mean()


def dice_np(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
	pred = (pred_mask > 0).astype(np.uint8)
	gt = (gt_mask > 0).astype(np.uint8)
	inter = float((pred & gt).sum())
	denom = float(pred.sum() + gt.sum())
	return float((2.0 * inter + eps) / (denom + eps))


class BCEDiceLoss(nn.Module):
	def __init__(self, bce_weight: float = 0.5):
		super().__init__()
		self.bce = nn.BCEWithLogitsLoss()
		self.bce_weight = bce_weight

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		bce = self.bce(logits, targets)
		# soft dice
		probs = torch.sigmoid(logits)
		targets = (targets > 0.5).float()
		dims = (1, 2, 3)
		intersection = (probs * targets).sum(dim=dims)
		union = probs.sum(dim=dims) + targets.sum(dim=dims)
		dice = 1 - ((2 * intersection + 1e-7) / (union + 1e-7)).mean()
		return self.bce_weight * bce + (1 - self.bce_weight) * dice


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	scaler: torch.cuda.amp.GradScaler | None,
	device: torch.device,
	criterion: nn.Module,
) -> float:
	model.train()
	losses = []
	for images, masks in tqdm(loader, desc="train", leave=False):
		images = images.to(device)
		masks = masks.to(device)
		optimizer.zero_grad(set_to_none=True)
		if scaler is not None:
			with torch.cuda.amp.autocast():
				logits = model(images)
				loss = criterion(logits, masks)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			logits = model(images)
			loss = criterion(logits, masks)
			loss.backward()
			optimizer.step()
		losses.append(loss.detach().item())
	return float(np.mean(losses))


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
	model.eval()
	losses = []
	dices = []
	for images, masks in tqdm(loader, desc="valid", leave=False):
		images = images.to(device)
		masks = masks.to(device)
		logits = model(images)
		loss = criterion(logits, masks)
		losses.append(loss.item())
		dices.append(dice_from_logits(logits, masks).item())
	return float(np.mean(losses)), float(np.mean(dices))


def list_images(folder: Path) -> list[Path]:
	exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
	paths = [p for p in folder.iterdir() if p.suffix.lower() in exts]
	return sorted(paths, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def run_train(args: argparse.Namespace) -> None:
	cfg = TrainConfig(
		data_root=Path(args.data_root),
		img_size=args.img_size,
		batch_size=args.batch_size,
		epochs=args.epochs,
		lr=args.lr,
		weight_decay=args.weight_decay,
		val_ratio=args.val_ratio,
		seed=args.seed,
		num_workers=args.num_workers,
		encoder=args.encoder,
		encoder_weights=args.encoder_weights,
		model=args.model,
		amp=not args.no_amp,
	)

	seed_everything(cfg.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_img_dir = cfg.data_root / "train" / "image"
	train_lbl_dir = cfg.data_root / "train" / "label"
	image_paths = list_images(train_img_dir)
	mask_paths = [train_lbl_dir / f"{p.stem}.jpg" for p in image_paths]
	missing = [p for p in mask_paths if not p.exists()]
	if missing:
		raise FileNotFoundError(f"Missing label files, e.g. {missing[0]}")

	idxs = list(range(len(image_paths)))
	train_idxs, val_idxs = train_test_split(
		idxs,
		test_size=cfg.val_ratio,
		random_state=cfg.seed,
		shuffle=True,
	)
	tr_images = [image_paths[i] for i in train_idxs]
	tr_masks = [mask_paths[i] for i in train_idxs]
	va_images = [image_paths[i] for i in val_idxs]
	va_masks = [mask_paths[i] for i in val_idxs]

	train_ds = FundusSegDataset(tr_images, tr_masks, build_augs(cfg.img_size, train=True))
	val_ds = FundusSegDataset(va_images, va_masks, build_augs(cfg.img_size, train=False))

	train_loader = DataLoader(
		train_ds,
		batch_size=cfg.batch_size,
		shuffle=True,
		num_workers=cfg.num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=max(1, cfg.batch_size),
		shuffle=False,
		num_workers=cfg.num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)

	model = build_model(cfg).to(device)
	criterion = BCEDiceLoss(bce_weight=0.5)
	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
	scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

	out_dir = Path(args.out_dir)
	ensure_dir(out_dir)
	best_path = out_dir / "best.pth"

	best_dice = -1.0
	for epoch in range(1, cfg.epochs + 1):
		tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
		va_loss, va_dice = validate(model, val_loader, device, criterion)
		print(
			f"epoch {epoch:03d}/{cfg.epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_dice={va_dice:.4f}"
		)
		if va_dice > best_dice:
			best_dice = va_dice
			torch.save(
				{
					"model": model.state_dict(),
					"cfg": cfg.__dict__,
					"best_dice": best_dice,
				},
				best_path,
			)
			print(f"saved best: {best_path} (dice={best_dice:.4f})")

	print(f"done. best dice={best_dice:.4f} at {best_path}")


def _train_with_explicit_split(
	cfg: TrainConfig,
	train_images: list[Path],
	train_masks: list[Path],
	val_images: list[Path],
	val_masks: list[Path],
	out_dir: Path,
) -> Path:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_ds = FundusSegDataset(train_images, train_masks, build_augs(cfg.img_size, train=True))
	val_ds = FundusSegDataset(val_images, val_masks, build_augs(cfg.img_size, train=False))
	train_loader = DataLoader(
		train_ds,
		batch_size=cfg.batch_size,
		shuffle=True,
		num_workers=cfg.num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=max(1, cfg.batch_size),
		shuffle=False,
		num_workers=cfg.num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)

	model = build_model(cfg).to(device)
	criterion = BCEDiceLoss(bce_weight=0.5)
	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
	scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

	ensure_dir(out_dir)
	best_path = out_dir / "best.pth"
	best_dice = -1.0
	for epoch in range(1, cfg.epochs + 1):
		tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
		va_loss, va_dice = validate(model, val_loader, device, criterion)
		print(
			f"  epoch {epoch:03d}/{cfg.epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_dice={va_dice:.4f}"
		)
		if va_dice > best_dice:
			best_dice = va_dice
			torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "best_dice": best_dice}, best_path)
	print(f"  fold best dice={best_dice:.4f} at {best_path}")
	return best_path


@torch.no_grad()
def predict_one(model: nn.Module, image: np.ndarray, device: torch.device, img_size: int) -> np.ndarray:
	import albumentations as A
	from albumentations.pytorch import ToTensorV2

	aug = A.Compose(
		[
			A.Resize(img_size, img_size, interpolation=1),
			A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			ToTensorV2(),
		]
	)
	x = aug(image=image)["image"].unsqueeze(0).to(device)
	logits = model(x)
	probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
	return probs


def probs_to_mask_image(probs: np.ndarray, thr: float) -> Image.Image:
	# Output must be grayscale where vessel pixels are >0.
	# We follow the official segmentation_to_csv.py which thresholds >0 as mask.
	mask = (probs > thr).astype(np.uint8) * 255
	return Image.fromarray(mask, mode="L")


def _sample_points_from_probs(
	probs: np.ndarray,
	n_pos: int,
	n_neg: int,
	pos_thr: float,
	neg_thr: float,
	seed: int,
):
	rng = np.random.default_rng(seed)
	pos = np.argwhere(probs >= pos_thr)
	neg = np.argwhere(probs <= neg_thr)

	points = []
	labels = []

	if len(pos) > 0 and n_pos > 0:
		choose = pos[rng.choice(len(pos), size=min(n_pos, len(pos)), replace=False)]
		for y, x in choose:
			points.append([float(x), float(y)])
			labels.append(1)

	if len(neg) > 0 and n_neg > 0:
		choose = neg[rng.choice(len(neg), size=min(n_neg, len(neg)), replace=False)]
		for y, x in choose:
			points.append([float(x), float(y)])
			labels.append(0)

	return points, labels


def _bbox_from_probs(probs: np.ndarray, thr: float) -> list[float] | None:
	ys, xs = np.where(probs >= thr)
	if len(xs) == 0:
		return None
	x1, x2 = int(xs.min()), int(xs.max())
	y1, y2 = int(ys.min()), int(ys.max())
	# xyxy
	return [float(x1), float(y1), float(x2), float(y2)]


def _largest_cc_mask(binary_255: np.ndarray) -> np.ndarray:
	"""Return a binary {0,1} mask for the largest connected component in a binary {0,255} image."""
	if binary_255.dtype != np.uint8:
		binary_255 = binary_255.astype(np.uint8)
	n, labels, stats, _ = cv2.connectedComponentsWithStats((binary_255 > 0).astype(np.uint8), connectivity=8)
	if n <= 1:
		return np.zeros_like(binary_255, dtype=np.uint8)
	# skip background at idx=0
	areas = stats[1:, cv2.CC_STAT_AREA]
	idx = int(np.argmax(areas)) + 1
	return (labels == idx).astype(np.uint8)


def _fill_holes(binary01: np.ndarray) -> np.ndarray:
	"""Fill holes in a {0,1} mask."""
	mask255 = (binary01 > 0).astype(np.uint8) * 255
	h, w = mask255.shape
	# flood fill from border on inverted mask to find background
	inv = (mask255 == 0).astype(np.uint8) * 255
	flood = inv.copy()
	ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
	cv2.floodFill(flood, ff_mask, (0, 0), 0)
	# pixels that remain 255 in flood are holes; fill them
	holes = (flood == 255)
	filled = mask255.copy()
	filled[holes] = 255
	return (filled > 0).astype(np.uint8)


def _estimate_fundus_mask(image_rgb: np.ndarray, margin_px: int = 10) -> np.ndarray:
	"""Estimate the fundus disk region as a {0,1} mask, optionally eroded by margin_px.

	This is used to remove obvious background/border noise while keeping vessels.
	"""
	gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	# If thresholding inverted, flip it.
	white_ratio = float(th.mean() / 255.0)
	if white_ratio > 0.6:
		th = cv2.bitwise_not(th)
	cc = _largest_cc_mask(th)
	if int(cc.sum()) == 0:
		return np.ones((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
	# smooth and fill holes
	k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
	cc255 = (cc * 255).astype(np.uint8)
	cc255 = cv2.morphologyEx(cc255, cv2.MORPH_CLOSE, k)
	cc = _fill_holes((cc255 > 0).astype(np.uint8))
	if margin_px > 0:
		k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
		cc = cv2.erode(cc.astype(np.uint8), k2)
	return (cc > 0).astype(np.uint8)


def _component_elongation(coords_yx: np.ndarray) -> float:
	"""Compute an elongation score from component coordinates (N,2) as y,x.

	Higher => more line-like.
	"""
	if coords_yx.shape[0] < 5:
		return 0.0
	xy = coords_yx[:, ::-1].astype(np.float32)  # to x,y
	xy -= xy.mean(axis=0, keepdims=True)
	cov = (xy.T @ xy) / max(1.0, float(xy.shape[0] - 1))
	# numerical safety
	try:
		w, _ = np.linalg.eigh(cov)
	except Exception:
		return 0.0
	w = np.sort(np.maximum(w, 1e-8))
	return float(w[-1] / w[0])


def _postprocess_vessel_mask(
	mask01: np.ndarray,
	image_rgb: np.ndarray,
	fundus_margin_px: int,
	border_px: int,
	frame_px: int,
	min_area_blob: int,
	min_area_line: int,
	elong_thr: float,
) -> np.ndarray:
	"""Remove obvious noise while preserving thin, line-like vessels."""
	mask01 = (mask01 > 0).astype(np.uint8)
	if int(mask01.sum()) == 0:
		return mask01
	# Hard-remove a thin frame at image borders.
	# Vessels should not exist on the literal image boundary; this fixes "white line" artifacts.
	if frame_px and frame_px > 0:
		fp = int(frame_px)
		mask01[:fp, :] = 0
		mask01[-fp:, :] = 0
		mask01[:, :fp] = 0
		mask01[:, -fp:] = 0
		if int(mask01.sum()) == 0:
			return mask01

	fundus = _estimate_fundus_mask(image_rgb, margin_px=fundus_margin_px)
	mask01 = (mask01 & fundus).astype(np.uint8)
	if int(mask01.sum()) == 0:
		return mask01

	# Distance to fundus boundary; small values mean "near border".
	# Use original (non-eroded) fundus for border distance.
	fundus_full = _estimate_fundus_mask(image_rgb, margin_px=0)
	dist = cv2.distanceTransform((fundus_full > 0).astype(np.uint8), cv2.DIST_L2, 3)

	n, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
	if n <= 1:
		return mask01

	areas = stats[1:, cv2.CC_STAT_AREA]
	main_id = int(np.argmax(areas)) + 1
	keep = np.zeros_like(mask01, dtype=np.uint8)

	for cid in range(1, n):
		area = int(stats[cid, cv2.CC_STAT_AREA])
		if area <= 0:
			continue
		coords = np.column_stack(np.where(labels == cid))  # y,x
		elong = _component_elongation(coords)
		near_border = bool(border_px > 0 and float(dist[labels == cid].min()) < float(border_px))

		# Keep rule:
		# - always keep main component
		# - keep sufficiently large blobs (main trunks)
		# - keep thin/line-like components even if small (fine vessels)
		# - if near border, require stronger line-likeness
		if cid == main_id:
			keep[labels == cid] = 1
			continue

		if area >= int(min_area_blob):
			keep[labels == cid] = 1
			continue

		if area >= int(min_area_line):
			thr = float(elong_thr * (1.5 if near_border else 1.0))
			if elong >= thr:
				keep[labels == cid] = 1
				continue

	return keep


def _combine_ultralytics_masks(result) -> np.ndarray:
	masks = getattr(result, "masks", None)
	if masks is None:
		return None
	data = getattr(masks, "data", None)
	if data is None:
		data = masks
	if isinstance(data, torch.Tensor):
		data = data.detach().cpu().numpy()
	if data is None:
		return None
	data = np.asarray(data)
	if data.ndim == 2:
		return (data > 0).astype(np.uint8)
	if data.ndim == 3:
		return (np.any(data > 0, axis=0)).astype(np.uint8)
	return None


def _sam_predict_sam_sam2(weights: str, image_rgb_512: np.ndarray, points, labels, bbox, device: str | None = None):
	from ultralytics import SAM

	model = SAM(weights)
	kwargs = {}
	if device is not None:
		kwargs["device"] = device
	bboxes = [bbox] if bbox is not None else None

	pts = points if points else None
	lbs = labels if labels else None
	# Ultralytics SAM expects points/labels to be grouped per object.
	# If a single bbox is given (one object) and we provide multiple points,
	# we must wrap them as one object: points shape (1, N, 2), labels shape (1, N).
	if bboxes is not None and pts is not None:
		pts = [pts]
		if lbs is None:
			lbs = [[1] * len(points)]
		else:
			lbs = [lbs]

	results = model.predict(
		source=image_rgb_512,
		points=pts,
		labels=lbs,
		bboxes=bboxes,
		# Ultralytics SAM uses a confidence filter even for interactive prompts.
		# The default can filter out all masks, returning masks.data with shape (0, H, W).
		# Setting conf=0.0 keeps prompt-based masks instead of producing all-black outputs.
		conf=0.0,
		verbose=False,
		**kwargs,
	)
	if not results:
		return None
	return _combine_ultralytics_masks(results[0])


def _resolve_ultralytics_weight(weight: str) -> str:
	"""Resolve a weight path.

	- If an existing path is given, return it.
	- If a bare filename is given (e.g. 'sam_b.pt'), let Ultralytics attempt to download it.
	"""
	weight = (weight or "").strip()
	if not weight:
		return ""
	p = Path(weight)
	if p.exists():
		return str(p)
	try:
		from ultralytics.utils.downloads import attempt_download_asset

		return str(attempt_download_asset(weight))
	except Exception as e:
		raise FileNotFoundError(
			f"Cannot find or download weight '{weight}'. Provide an absolute/relative path to a local .pt/.pth file. ({type(e).__name__}: {e})"
		)


def _sam3_semantic_predict(weights: str, image_rgb_512: np.ndarray, bbox, text_list: list[str], device: str | None = None):
	# Uses Ultralytics SAM3SemanticPredictor directly since ultralytics.SAM wrapper maps sam3->SAM3Predictor (interactive),
	# while text-based segmentation is implemented in SAM3SemanticPredictor.
	from ultralytics.models.sam.predict import SAM3SemanticPredictor

	overrides = {
		"model": weights,
		"task": "segment",
		"mode": "predict",
		"imgsz": 1024,
		"conf": 0.25,
	}
	if device is not None:
		overrides["device"] = device
	predictor = SAM3SemanticPredictor(overrides=overrides)
	# Provide bbox to focus region; SAM3SemanticPredictor doesn't use point prompts.
	bboxes = [bbox] if bbox is not None else None
	labels = [1] if bbox is not None else None
	results = predictor(source=image_rgb_512, stream=False, bboxes=bboxes, labels=labels, text=text_list)
	if not results:
		return None
	return _combine_ultralytics_masks(results[0])


def run_compare_sam(args: argparse.Namespace) -> None:
	seed_everything(args.seed)

	data_root = Path(args.data_root)
	train_img_dir = data_root / "train" / "image"
	train_lbl_dir = data_root / "train" / "label"
	image_paths = list_images(train_img_dir)
	mask_paths = [train_lbl_dir / f"{p.stem}.jpg" for p in image_paths]
	missing = [p for p in mask_paths if not p.exists()]
	if missing:
		raise FileNotFoundError(f"Missing label files, e.g. {missing[0]}")

	out_dir = Path(args.out_dir)
	ensure_dir(out_dir)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"device: {device}")

	kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
	all_scores = {
		"unet": [],
		"sam1": [],
		"sam2": [],
		"sam3": [],
	}

	sam1_weights = _resolve_ultralytics_weight(args.sam1_weights)
	sam2_weights = _resolve_ultralytics_weight(args.sam2_weights)
	sam3_weights = _resolve_ultralytics_weight(args.sam3_weights)
	sam3_text = [args.sam3_text]

	for fold, (tr_idx, va_idx) in enumerate(kf.split(image_paths), start=1):
		print(f"\n=== fold {fold}/{args.folds} ===")
		tr_images = [image_paths[i] for i in tr_idx]
		tr_masks = [mask_paths[i] for i in tr_idx]
		va_images = [image_paths[i] for i in va_idx]
		va_masks = [mask_paths[i] for i in va_idx]

		cfg = TrainConfig(
			data_root=data_root,
			img_size=args.img_size,
			batch_size=args.batch_size,
			epochs=args.epochs,
			lr=args.lr,
			weight_decay=args.weight_decay,
			val_ratio=0.0,
			seed=args.seed + fold,
			num_workers=args.num_workers,
			encoder=args.encoder,
			encoder_weights=args.encoder_weights,
			model=args.model,
			amp=not args.no_amp,
		)
		fold_dir = out_dir / f"fold_{fold}"
		best_ckpt = _train_with_explicit_split(cfg, tr_images, tr_masks, va_images, va_masks, fold_dir)

		ckpt = torch.load(best_ckpt, map_location="cpu")
		model = build_model(cfg)
		model.load_state_dict(ckpt["model"], strict=True)
		model.to(torch.device(device))
		model.eval()

		fold_scores = {"unet": [], "sam1": [], "sam2": [], "sam3": []}
		for img_path, lbl_path in tqdm(list(zip(va_images, va_masks)), desc=f"eval fold {fold}"):
			# UNet probs are produced on cfg.img_size, so we run SAM on the same resized image for coordinate consistency.
			img_rgb = load_rgb(img_path)
			img_rgb_512 = np.array(Image.fromarray(img_rgb).resize((cfg.img_size, cfg.img_size), Image.Resampling.BILINEAR))
			probs = predict_one(model, img_rgb, torch.device(device), cfg.img_size)

			gt = load_mask(lbl_path)
			gt_512 = resize_mask_nearest(gt, cfg.img_size)

			unet_mask = (probs > args.unet_threshold).astype(np.uint8)
			fold_scores["unet"].append(dice_np(unet_mask, gt_512))

			# fixed prompts derived only from UNet probs
			bbox = _bbox_from_probs(probs, thr=args.bbox_threshold)
			points, point_labels = _sample_points_from_probs(
				probs,
				n_pos=args.n_pos,
				n_neg=args.n_neg,
				pos_thr=args.pos_thr,
				neg_thr=args.neg_thr,
				seed=(args.seed * 1000 + fold * 100 + int(img_path.stem)),
			)

			if sam1_weights:
				try:
					sam_mask = _sam_predict_sam_sam2(sam1_weights, img_rgb_512, points, point_labels, bbox, device=device)
					if sam_mask is None:
						fold_scores["sam1"].append(0.0)
					else:
						fold_scores["sam1"].append(dice_np(sam_mask, gt_512))
				except Exception as e:
					raise RuntimeError(f"SAM1 failed on {img_path.name}: {e}")

			if sam2_weights:
				try:
					sam_mask = _sam_predict_sam_sam2(sam2_weights, img_rgb_512, points, point_labels, bbox, device=device)
					if sam_mask is None:
						fold_scores["sam2"].append(0.0)
					else:
						fold_scores["sam2"].append(dice_np(sam_mask, gt_512))
				except Exception as e:
					raise RuntimeError(f"SAM2 failed on {img_path.name}: {e}")

			if sam3_weights:
				try:
					sam_mask = _sam3_semantic_predict(sam3_weights, img_rgb_512, bbox, sam3_text, device=device)
					if sam_mask is None:
						fold_scores["sam3"].append(0.0)
					else:
						fold_scores["sam3"].append(dice_np(sam_mask, gt_512))
				except Exception as e:
					raise RuntimeError(f"SAM3 failed on {img_path.name}: {e}")

		for k in fold_scores:
			if len(fold_scores[k]) > 0:
				m = float(np.mean(fold_scores[k]))
				print(f"fold {fold} {k}: mean dice={m:.4f}")
				all_scores[k].append(m)

	print("\n=== summary (mean over folds) ===")
	for k, vals in all_scores.items():
		if len(vals) == 0:
			continue
		print(f"{k}: {float(np.mean(vals)):.4f} ± {float(np.std(vals)):.4f} over {len(vals)} folds")


def run_refine_sam(args: argparse.Namespace) -> None:
	"""Run SAM refinement on a split (train/test) using UNet-derived auto prompts."""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"device: {device}")

	data_root = Path(args.data_root)
	split = args.split
	img_dir = data_root / split / "image"
	if not img_dir.exists():
		raise FileNotFoundError(f"missing: {img_dir}")

	image_paths = list_images(img_dir)
	if len(image_paths) == 0:
		raise FileNotFoundError(f"no images found under: {img_dir}")

	# Load UNet checkpoint
	ckpt = torch.load(args.unet_checkpoint, map_location="cpu")
	cfg_dict = ckpt.get("cfg", {})
	cfg = TrainConfig(
		data_root=data_root,
		img_size=int(cfg_dict.get("img_size", args.img_size)),
		encoder=str(cfg_dict.get("encoder", args.encoder)),
		encoder_weights=str(cfg_dict.get("encoder_weights", args.encoder_weights)),
		model=str(cfg_dict.get("model", args.model)),
	)
	unet = build_model(cfg)
	unet.load_state_dict(ckpt["model"], strict=True)
	unet.to(torch.device(device))
	unet.eval()

	sam_variant = args.sam_variant.lower()
	sam_weights = _resolve_ultralytics_weight(args.sam_weights)
	if not sam_weights:
		raise ValueError("--sam_weights is required")

	out_dir = Path(args.out_dir)
	ensure_dir(out_dir)

	for p in tqdm(image_paths, desc=f"refine_{sam_variant}"):
		img_rgb = load_rgb(p)
		img_rgb_512 = np.array(
			Image.fromarray(img_rgb).resize((cfg.img_size, cfg.img_size), Image.Resampling.BILINEAR)
		)
		probs = predict_one(unet, img_rgb, torch.device(device), cfg.img_size)
		unet_mask_fallback = (probs > args.unet_threshold).astype(np.uint8)

		bbox = _bbox_from_probs(probs, thr=args.bbox_threshold) if args.use_bbox else None
		points, point_labels = _sample_points_from_probs(
			probs,
			n_pos=args.n_pos,
			n_neg=args.n_neg,
			pos_thr=args.pos_thr,
			neg_thr=args.neg_thr,
			seed=(args.seed * 1000 + int(p.stem) if p.stem.isdigit() else args.seed),
		)

		if sam_variant in {"sam1", "sam", "sam2"}:
			mask = _sam_predict_sam_sam2(
				sam_weights,
				img_rgb_512,
				points,
				point_labels,
				bbox,
				device=device,
			)
		elif sam_variant in {"sam3"}:
			mask = _sam3_semantic_predict(
				sam_weights,
				img_rgb_512,
				bbox,
				[args.sam3_text],
				device=device,
			)
		else:
			raise ValueError("--sam_variant must be one of: sam1, sam2, sam3")

		# If SAM returns no mask (or an empty mask), fall back to the UNet prediction.
		# This prevents the refinement stage from wiping out vessels.
		if mask is None or int(np.asarray(mask).sum()) == 0:
			mask = unet_mask_fallback
		else:
			mask = (np.asarray(mask) > 0).astype(np.uint8)
			# Vessel segmentation is a "thin structure" task. SAM can easily snap to the
			# large fundus disk when prompted with a broad bbox. To prevent over-segmentation
			# ("white eyeball"), constrain SAM output using a UNet probability prior.
			if args.combine != "sam":
				prior = (probs > args.prior_thr).astype(np.uint8)
				if args.combine == "and":
					mask = (mask & prior).astype(np.uint8)
				elif args.combine == "or":
					mask = (mask | unet_mask_fallback).astype(np.uint8)
				else:
					raise ValueError("--combine must be one of: sam, and, or")
			# If constraint makes it empty, fall back.
			if int(mask.sum()) == 0:
				mask = unet_mask_fallback
		# Optional post-process: remove background/border noise while preserving thin vessels.
		if args.postprocess:
			mask_pp = _postprocess_vessel_mask(
				mask,
				img_rgb_512,
				fundus_margin_px=args.fundus_margin_px,
				border_px=args.border_px,
				frame_px=args.frame_px,
				min_area_blob=args.min_area_blob,
				min_area_line=args.min_area_line,
				elong_thr=args.elong_thr,
			)
			if int(mask_pp.sum()) > 0:
				mask = mask_pp
		mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
		mask_img.save(out_dir / f"{p.stem}.png")

	print(f"saved {len(image_paths)} refined masks to: {out_dir}")


def run_predict(args: argparse.Namespace) -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ckpt = torch.load(args.checkpoint, map_location="cpu")
	cfg_dict = ckpt.get("cfg", {})

	# Build model from either CLI args or checkpoint config
	cfg = TrainConfig(
		data_root=Path(args.data_root),
		img_size=int(cfg_dict.get("img_size", args.img_size)),
		encoder=str(cfg_dict.get("encoder", args.encoder)),
		encoder_weights=str(cfg_dict.get("encoder_weights", args.encoder_weights)),
		model=str(cfg_dict.get("model", args.model)),
	)
	model = build_model(cfg)
	model.load_state_dict(ckpt["model"], strict=True)
	model.to(device)
	model.eval()

	test_img_dir = Path(args.data_root) / "test" / "image"
	image_paths = list_images(test_img_dir)

	out_dir = Path(args.out_dir)
	ensure_dir(out_dir)

	for p in tqdm(image_paths, desc="predict"):
		img = load_rgb(p)
		probs = predict_one(model, img, device, cfg.img_size)
		mask_img = probs_to_mask_image(probs, args.threshold)
		# Save with same stem; png avoids jpeg artifacts
		mask_img.save(out_dir / f"{p.stem}.png")

	print(f"saved {len(image_paths)} masks to: {out_dir}")


def run_to_csv(args: argparse.Namespace) -> None:
	# Call the provided converter script; it expects ./image under segmentation directory.
	seg_dir = Path(args.data_root).resolve()
	converter = (seg_dir / "segmentation_to_csv.py").resolve()
	if not converter.exists():
		raise FileNotFoundError(f"missing: {converter}")
	cwd = os.getcwd()
	try:
		os.chdir(seg_dir)
		import runpy

		runpy.run_path(str(converter), run_name="__main__")
		print(f"wrote: {seg_dir / 'submission.csv'}")
	finally:
		os.chdir(cwd)


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Task5 vessel segmentation (train/predict/to_csv)")
	sub = p.add_subparsers(dest="cmd", required=True)

	p_tr = sub.add_parser("train", help="train UNet-like model")
	p_tr.add_argument("--data_root", type=str, default="NeuMachineLearning-main/task5/segmentation")
	p_tr.add_argument("--out_dir", type=str, default="NeuMachineLearning-main/task5/outputs")
	p_tr.add_argument("--img_size", type=int, default=512)
	p_tr.add_argument("--batch_size", type=int, default=4)
	p_tr.add_argument("--epochs", type=int, default=40)
	p_tr.add_argument("--lr", type=float, default=3e-4)
	p_tr.add_argument("--weight_decay", type=float, default=1e-4)
	p_tr.add_argument("--val_ratio", type=float, default=0.2)
	p_tr.add_argument("--seed", type=int, default=42)
	p_tr.add_argument("--num_workers", type=int, default=2)
	p_tr.add_argument("--model", type=str, default="unetplusplus", choices=["unet", "unetplusplus"])
	p_tr.add_argument("--encoder", type=str, default="resnet34")
	p_tr.add_argument("--encoder_weights", type=str, default="imagenet")
	p_tr.add_argument("--no_amp", action="store_true")
	p_tr.set_defaults(func=run_train)

	p_pr = sub.add_parser("predict", help="predict test masks")
	p_pr.add_argument("--data_root", type=str, default="NeuMachineLearning-main/task5/segmentation")
	p_pr.add_argument("--checkpoint", type=str, required=True)
	p_pr.add_argument("--out_dir", type=str, default="NeuMachineLearning-main/task5/segmentation/image")
	p_pr.add_argument("--img_size", type=int, default=512)
	p_pr.add_argument("--threshold", type=float, default=0.5)
	p_pr.add_argument("--model", type=str, default="unetplusplus", choices=["unet", "unetplusplus"])
	p_pr.add_argument("--encoder", type=str, default="resnet34")
	p_pr.add_argument("--encoder_weights", type=str, default="imagenet")
	p_pr.set_defaults(func=run_predict)

	p_csv = sub.add_parser("to_csv", help="run official segmentation_to_csv.py")
	p_csv.add_argument("--data_root", type=str, default="NeuMachineLearning-main/task5/segmentation")
	p_csv.set_defaults(func=run_to_csv)

	p_cmp = sub.add_parser("compare_sam", help="5-fold compare UNet vs SAM1/SAM2/SAM3 refiners")
	p_cmp.add_argument("--data_root", type=str, default="NeuMachineLearning-main/task5/segmentation")
	p_cmp.add_argument("--out_dir", type=str, default="NeuMachineLearning-main/task5/compare_sam")
	p_cmp.add_argument("--folds", type=int, default=5)
	p_cmp.add_argument("--img_size", type=int, default=512)
	p_cmp.add_argument("--batch_size", type=int, default=2)
	p_cmp.add_argument("--epochs", type=int, default=6)
	p_cmp.add_argument("--lr", type=float, default=3e-4)
	p_cmp.add_argument("--weight_decay", type=float, default=1e-4)
	p_cmp.add_argument("--seed", type=int, default=42)
	p_cmp.add_argument("--num_workers", type=int, default=2)
	p_cmp.add_argument("--model", type=str, default="unetplusplus", choices=["unet", "unetplusplus"])
	p_cmp.add_argument("--encoder", type=str, default="resnet34")
	p_cmp.add_argument("--encoder_weights", type=str, default="imagenet")
	p_cmp.add_argument("--no_amp", action="store_true")

	# UNet->mask and prompt extraction
	p_cmp.add_argument("--unet_threshold", type=float, default=0.5)
	p_cmp.add_argument("--bbox_threshold", type=float, default=0.6)
	p_cmp.add_argument("--pos_thr", type=float, default=0.7)
	p_cmp.add_argument("--neg_thr", type=float, default=0.2)
	p_cmp.add_argument("--n_pos", type=int, default=16)
	p_cmp.add_argument("--n_neg", type=int, default=16)

	# SAM weights (Ultralytics .pt files). If omitted, that backend is skipped.
	p_cmp.add_argument("--sam1_weights", type=str, default="")
	p_cmp.add_argument("--sam2_weights", type=str, default="")
	p_cmp.add_argument("--sam3_weights", type=str, default="")
	p_cmp.add_argument("--sam3_text", type=str, default="blood vessel")
	p_cmp.set_defaults(func=run_compare_sam)

	p_ref = sub.add_parser("refine_sam", help="refine UNet prediction with SAM using auto-prompts")
	p_ref.add_argument("--data_root", type=str, default="NeuMachineLearning-main/task5/segmentation")
	p_ref.add_argument("--split", type=str, default="test", choices=["train", "test"])
	p_ref.add_argument("--unet_checkpoint", type=str, required=True)
	p_ref.add_argument("--out_dir", type=str, default="NeuMachineLearning-main/task5/segmentation/image")

	p_ref.add_argument("--img_size", type=int, default=512)
	p_ref.add_argument("--model", type=str, default="unetplusplus", choices=["unet", "unetplusplus"])
	p_ref.add_argument("--encoder", type=str, default="resnet34")
	p_ref.add_argument("--encoder_weights", type=str, default="imagenet")

	p_ref.add_argument("--sam_variant", type=str, default="sam2", choices=["sam1", "sam2", "sam3"])
	p_ref.add_argument("--sam_weights", type=str, required=True)
	p_ref.add_argument("--sam3_text", type=str, default="blood vessel")

	p_ref.add_argument("--seed", type=int, default=42)
	p_ref.add_argument("--unet_threshold", type=float, default=0.5)
	p_ref.add_argument(
		"--combine",
		type=str,
		default="and",
		choices=["sam", "and", "or"],
		help="How to combine SAM result with UNet prior. 'and' prevents fundus-disk over-segmentation.",
	)
	p_ref.add_argument(
		"--prior_thr",
		type=float,
		default=0.3,
		help="UNet probability threshold used as prior when --combine=and.",
	)
	p_ref.add_argument(
		"--use_bbox",
		action="store_true",
		help="Use UNet-derived bbox as SAM prompt. Often hurts vessel segmentation; default off.",
	)
	p_ref.add_argument(
		"--postprocess",
		action="store_true",
		help="Enable noise suppression postprocess (fundus masking + component filtering).",
	)
	p_ref.add_argument("--fundus_margin_px", type=int, default=12, help="Erode fundus mask by this many pixels.")
	p_ref.add_argument("--border_px", type=int, default=14, help="Border band width (pixels) to treat as periphery.")
	p_ref.add_argument(
		"--frame_px",
		type=int,
		default=6,
		help="Force-clear this many pixels at the image border (removes border white-line artifacts).",
	)
	p_ref.add_argument("--min_area_blob", type=int, default=120, help="Keep components with area >= this (pixels).")
	p_ref.add_argument(
		"--min_area_line",
		type=int,
		default=25,
		help="Minimum area for a line-like component to be considered (pixels).",
	)
	p_ref.add_argument(
		"--elong_thr",
		type=float,
		default=10.0,
		help="Elongation threshold (higher keeps only more line-like components).",
	)
	p_ref.add_argument("--bbox_threshold", type=float, default=0.6)
	p_ref.add_argument("--pos_thr", type=float, default=0.7)
	p_ref.add_argument("--neg_thr", type=float, default=0.2)
	p_ref.add_argument("--n_pos", type=int, default=16)
	p_ref.add_argument("--n_neg", type=int, default=16)
	p_ref.set_defaults(func=run_refine_sam)

	return p


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
