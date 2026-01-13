import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
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


def load_rgb(path: Path) -> np.ndarray:
	return np.array(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
	arr = np.array(Image.open(path).convert("L"))
	return (arr > 0).astype(np.uint8)


def resize_mask_nearest(mask: np.ndarray, size: int) -> np.ndarray:
	if mask.shape[0] == size and mask.shape[1] == size:
		return mask
	return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)


def list_images(folder: Path) -> list[Path]:
	exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
	paths = [p for p in folder.iterdir() if p.suffix.lower() in exts]
	return sorted(paths, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


@dataclass
class TrainConfig:
	data_root: Path
	img_size: int = 512
	batch_size: int = 2
	epochs: int = 80
	lr: float = 3e-4
	min_lr: float = 1e-6
	weight_decay: float = 1e-4
	seed: int = 42
	num_workers: int = 2
	encoder: str = "efficientnet-b3"
	encoder_weights: str = "imagenet"
	model: str = "unetplusplus"  # unet / unetplusplus / manet / fpn / pan / deeplabv3plus
	amp: bool = True
	loss: str = "focal_dice"  # bce_dice / focal_dice


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
			return transformed["image"]

		transformed = self.aug(image=image, mask=mask)
		image_t = transformed["image"]
		mask_t = transformed["mask"]
		if isinstance(mask_t, np.ndarray):
			mask_t = torch.from_numpy(mask_t)
		mask_t = mask_t.float()
		if mask_t.ndim == 2:
			mask_t = mask_t.unsqueeze(0)
		return image_t, mask_t


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
					shift_limit=0.06,
					scale_limit=0.20,
					rotate_limit=20,
					border_mode=0,
					p=0.6,
				),
				A.OneOf(
					[
						A.RandomBrightnessContrast(p=1.0),
						A.HueSaturationValue(p=1.0),
						A.RandomGamma(p=1.0),
					],
					p=0.8,
				),
				A.OneOf(
					[
						A.GaussianBlur(blur_limit=(3, 5), p=1.0),
						A.MotionBlur(blur_limit=5, p=1.0),
						A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
					],
					p=0.25,
				),
				A.CLAHE(p=0.2),
				A.GridDistortion(p=0.15),
				A.ElasticTransform(alpha=20, sigma=5, alpha_affine=5, p=0.1),
				A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.2),
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


def build_model(cfg: TrainConfig) -> nn.Module:
	import segmentation_models_pytorch as smp

	name = cfg.model.lower()
	kwargs = dict(
		encoder_name=cfg.encoder,
		encoder_weights=cfg.encoder_weights,
		in_channels=3,
		classes=1,
	)
	if name in {"unet", "u-net"}:
		return smp.Unet(**kwargs)
	if name in {"unetplusplus", "unet++", "unet_plus_plus"}:
		return smp.UnetPlusPlus(**kwargs)
	if name in {"manet", "ma-net", "ma_net"}:
		return smp.MAnet(**kwargs)
	if name in {"fpn"}:
		return smp.FPN(**kwargs)
	if name in {"pan"}:
		return smp.PAN(**kwargs)
	if name in {"deeplabv3plus", "deeplabv3+", "dlv3+"}:
		return smp.DeepLabV3Plus(**kwargs)
	raise ValueError(f"Unknown model: {cfg.model}")


def soft_dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
	probs = torch.sigmoid(logits)
	targets = (targets > 0.5).float()
	dims = (1, 2, 3)
	intersection = (probs * targets).sum(dim=dims)
	union = probs.sum(dim=dims) + targets.sum(dim=dims)
	dice = (2 * intersection + eps) / (union + eps)
	return 1.0 - dice.mean()


def dice_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
	probs = torch.sigmoid(logits)
	preds = (probs > 0.5).float()
	targets = (targets > 0.5).float()
	intersection = (preds * targets).sum(dim=(1, 2, 3))
	union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
	return ((2 * intersection + eps) / (union + eps)).mean()


class FocalLossWithLogits(nn.Module):
	def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
		super().__init__()
		self.gamma = gamma
		self.alpha = alpha

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		targets = (targets > 0.5).float()
		bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
		p = torch.sigmoid(logits)
		pt = torch.where(targets > 0.5, p, 1 - p)
		alpha_t = torch.where(targets > 0.5, torch.full_like(pt, self.alpha), torch.full_like(pt, 1 - self.alpha))
		loss = alpha_t * ((1 - pt) ** self.gamma) * bce
		return loss.mean()


class CombinedLoss(nn.Module):
	def __init__(self, kind: str):
		super().__init__()
		self.kind = kind
		self.bce = nn.BCEWithLogitsLoss()
		self.focal = FocalLossWithLogits()

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		if self.kind == "bce_dice":
			return 0.5 * self.bce(logits, targets) + 0.5 * soft_dice_loss_from_logits(logits, targets)
		if self.kind == "focal_dice":
			return 0.5 * self.focal(logits, targets) + 0.5 * soft_dice_loss_from_logits(logits, targets)
		raise ValueError("--loss must be one of: bce_dice, focal_dice")


@torch.no_grad()
def predict_probs(model: nn.Module, image_rgb: np.ndarray, device: torch.device, img_size: int) -> np.ndarray:
	import albumentations as A
	from albumentations.pytorch import ToTensorV2

	aug = A.Compose(
		[
			A.Resize(img_size, img_size, interpolation=1),
			A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			ToTensorV2(),
		]
	)
	x = aug(image=image_rgb)["image"].unsqueeze(0).to(device)
	logits = model(x)
	probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
	return probs


def probs_to_mask_image(probs: np.ndarray, thr: float) -> Image.Image:
	mask = (probs > thr).astype(np.uint8) * 255
	return Image.fromarray(mask, mode="L")


def _train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	scaler: torch.cuda.amp.GradScaler | None,
	device: torch.device,
	criterion: nn.Module,
) -> float:
	model.train()
	losses: list[float] = []
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
		losses.append(float(loss.detach().item()))
	return float(np.mean(losses))


@torch.no_grad()
def _validate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
	model.eval()
	losses: list[float] = []
	dices: list[float] = []
	for images, masks in tqdm(loader, desc="valid", leave=False):
		images = images.to(device)
		masks = masks.to(device)
		logits = model(images)
		loss = criterion(logits, masks)
		losses.append(float(loss.item()))
		dices.append(float(dice_from_logits(logits, masks).item()))
	return float(np.mean(losses)), float(np.mean(dices))


def run_train_cv(args: argparse.Namespace) -> None:
	cfg = TrainConfig(
		data_root=Path(args.data_root),
		img_size=args.img_size,
		batch_size=args.batch_size,
		epochs=args.epochs,
		lr=args.lr,
		min_lr=args.min_lr,
		weight_decay=args.weight_decay,
		seed=args.seed,
		num_workers=args.num_workers,
		encoder=args.encoder,
		encoder_weights=args.encoder_weights,
		model=args.model,
		amp=not args.no_amp,
		loss=args.loss,
	)
	seed_everything(cfg.seed)

	data_root = cfg.data_root
	train_img_dir = data_root / "train" / "image"
	train_lbl_dir = data_root / "train" / "label"
	image_paths = list_images(train_img_dir)
	mask_paths = [train_lbl_dir / f"{p.stem}.jpg" for p in image_paths]
	missing = [p for p in mask_paths if not p.exists()]
	if missing:
		raise FileNotFoundError(f"Missing label files, e.g. {missing[0]}")

	out_dir = Path(args.out_dir)
	ensure_dir(out_dir)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"device: {device}")

	kf = KFold(n_splits=args.folds, shuffle=True, random_state=cfg.seed)
	for fold, (tr_idx, va_idx) in enumerate(kf.split(image_paths), start=1):
		fold_dir = out_dir / f"fold_{fold}"
		ensure_dir(fold_dir)
		print(f"\n=== fold {fold}/{args.folds} ===")

		tr_images = [image_paths[i] for i in tr_idx]
		tr_masks = [mask_paths[i] for i in tr_idx]
		va_images = [image_paths[i] for i in va_idx]
		va_masks = [mask_paths[i] for i in va_idx]

		train_ds = FundusSegDataset(tr_images, tr_masks, build_augs(cfg.img_size, train=True))
		val_ds = FundusSegDataset(va_images, va_masks, build_augs(cfg.img_size, train=False))

		train_loader = DataLoader(
			train_ds,
			batch_size=cfg.batch_size,
			shuffle=True,
			num_workers=cfg.num_workers,
			pin_memory=True,
			drop_last=False,
		)
		val_loader = DataLoader(
			val_ds,
			batch_size=max(1, cfg.batch_size),
			shuffle=False,
			num_workers=cfg.num_workers,
			pin_memory=True,
			drop_last=False,
		)

		model = build_model(cfg).to(device)
		criterion = CombinedLoss(cfg.loss)
		optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
		scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

		best_dice = -1.0
		best_path = fold_dir / "best.pth"

		for epoch in range(1, cfg.epochs + 1):
			train_loss = _train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
			val_loss, val_dice = _validate(model, val_loader, device, criterion)
			scheduler.step()
			lr_now = float(optimizer.param_groups[0]["lr"])
			print(
				f"epoch {epoch:03d}/{cfg.epochs} | lr={lr_now:.2e} | train={train_loss:.4f} | val={val_loss:.4f} | dice={val_dice:.4f}"
			)
			if val_dice > best_dice:
				best_dice = val_dice
				payload = {
					"model": model.state_dict(),
					"cfg": {
						"img_size": cfg.img_size,
						"encoder": cfg.encoder,
						"encoder_weights": cfg.encoder_weights,
						"model": cfg.model,
						"loss": cfg.loss,
					},
				}
				torch.save(payload, best_path)

		print(f"fold {fold} best dice: {best_dice:.4f} -> {best_path}")

	print(f"done. checkpoints: {out_dir}")


def _load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[nn.Module, TrainConfig]:
	ckpt = torch.load(ckpt_path, map_location="cpu")
	cfg_dict = ckpt.get("cfg", {})
	cfg = TrainConfig(
		data_root=Path("."),
		img_size=int(cfg_dict.get("img_size", 512)),
		encoder=str(cfg_dict.get("encoder", "efficientnet-b3")),
		encoder_weights=str(cfg_dict.get("encoder_weights", "imagenet")),
		model=str(cfg_dict.get("model", "unetplusplus")),
		loss=str(cfg_dict.get("loss", "focal_dice")),
	)
	model = build_model(cfg)
	model.load_state_dict(ckpt["model"], strict=True)
	model.to(device)
	model.eval()
	return model, cfg


@torch.no_grad()
def _predict_probs_tta(model: nn.Module, image_rgb: np.ndarray, device: torch.device, img_size: int, tta: bool) -> np.ndarray:
	if not tta:
		return predict_probs(model, image_rgb, device, img_size)

	# Simple flip TTA: identity + hflip + vflip
	probs = []
	p0 = predict_probs(model, image_rgb, device, img_size)
	probs.append(p0)
	# hflip
	img_h = np.ascontiguousarray(image_rgb[:, ::-1, :])
	p_h = predict_probs(model, img_h, device, img_size)[:, ::-1]
	probs.append(p_h)
	# vflip
	img_v = np.ascontiguousarray(image_rgb[::-1, :, :])
	p_v = predict_probs(model, img_v, device, img_size)[::-1, :]
	probs.append(p_v)

	return np.mean(np.stack(probs, axis=0), axis=0)


def run_predict_ensemble(args: argparse.Namespace) -> None:
	data_root = Path(args.data_root)
	split = args.split
	img_dir = data_root / split / "image"
	image_paths = list_images(img_dir)
	if not image_paths:
		raise FileNotFoundError(f"no images under: {img_dir}")

	ckpt_root = Path(args.ckpt_root)
	ckpts = []
	for i in range(1, args.folds + 1):
		p = ckpt_root / f"fold_{i}" / "best.pth"
		if not p.exists():
			raise FileNotFoundError(f"missing fold checkpoint: {p}")
		ckpts.append(p)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"device: {device}")
	models = []
	img_size = int(args.img_size)
	for p in ckpts:
		m, cfg = _load_model_from_ckpt(p, device)
		models.append(m)
		img_size = int(cfg.img_size)

	out_dir = Path(args.out_dir)
	ensure_dir(out_dir)

	for p in tqdm(image_paths, desc=f"predict_ens_{split}"):
		img = load_rgb(p)
		probs_all = []
		for m in models:
			probs_all.append(_predict_probs_tta(m, img, device, img_size, tta=args.tta))
		probs = np.mean(np.stack(probs_all, axis=0), axis=0)
		mask_img = probs_to_mask_image(probs, args.threshold)
		mask_img.save(out_dir / f"{p.stem}.png")

	print(f"saved {len(image_paths)} masks to: {out_dir}")


def run_to_csv(args: argparse.Namespace) -> None:
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
	p = argparse.ArgumentParser(description="Task5 vessel segmentation (v2: stronger baselines + CV ensemble)")
	sub = p.add_subparsers(dest="cmd", required=True)

	p_tr = sub.add_parser("train_cv", help="K-fold train with stronger augs/loss")
	p_tr.add_argument("--data_root", type=str, default="NeuMachineLearning-main/task5/segmentation")
	p_tr.add_argument("--out_dir", type=str, default="NeuMachineLearning-main/task5/outputs_v2")
	p_tr.add_argument("--folds", type=int, default=5)
	p_tr.add_argument("--img_size", type=int, default=512)
	p_tr.add_argument("--batch_size", type=int, default=2)
	p_tr.add_argument("--epochs", type=int, default=80)
	p_tr.add_argument("--lr", type=float, default=3e-4)
	p_tr.add_argument("--min_lr", type=float, default=1e-6)
	p_tr.add_argument("--weight_decay", type=float, default=1e-4)
	p_tr.add_argument("--seed", type=int, default=42)
	p_tr.add_argument("--num_workers", type=int, default=2)
	p_tr.add_argument(
		"--model",
		type=str,
		default="unetplusplus",
		choices=["unet", "unetplusplus", "manet", "fpn", "pan", "deeplabv3plus"],
	)
	p_tr.add_argument("--encoder", type=str, default="efficientnet-b3")
	p_tr.add_argument("--encoder_weights", type=str, default="imagenet")
	p_tr.add_argument("--loss", type=str, default="focal_dice", choices=["bce_dice", "focal_dice"])
	p_tr.add_argument("--no_amp", action="store_true")
	p_tr.set_defaults(func=run_train_cv)

	p_pr = sub.add_parser("predict_ensemble", help="predict using CV ensemble (optionally with TTA)")
	p_pr.add_argument("--data_root", type=str, default="NeuMachineLearning-main/task5/segmentation")
	p_pr.add_argument("--split", type=str, default="test", choices=["train", "test"])
	p_pr.add_argument("--ckpt_root", type=str, required=True, help="folder containing fold_1/best.pth ...")
	p_pr.add_argument("--folds", type=int, default=5)
	p_pr.add_argument("--out_dir", type=str, default="NeuMachineLearning-main/task5/segmentation/image")
	p_pr.add_argument("--img_size", type=int, default=512)
	p_pr.add_argument("--threshold", type=float, default=0.45)
	p_pr.add_argument("--tta", action="store_true")
	p_pr.set_defaults(func=run_predict_ensemble)

	p_csv = sub.add_parser("to_csv", help="run official segmentation_to_csv.py")
	p_csv.add_argument("--data_root", type=str, default="NeuMachineLearning-main/task5/segmentation")
	p_csv.set_defaults(func=run_to_csv)

	return p


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
