import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
	return np.array(Image.open(path).convert("RGB"))


def list_images(folder: Path) -> list[Path]:
	exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
	paths = [p for p in folder.iterdir() if p.suffix.lower() in exts]
	return sorted(paths, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def _largest_cc_mask(binary_255: np.ndarray) -> np.ndarray:
	if binary_255.dtype != np.uint8:
		binary_255 = binary_255.astype(np.uint8)
	n, labels, stats, _ = cv2.connectedComponentsWithStats((binary_255 > 0).astype(np.uint8), connectivity=8)
	if n <= 1:
		return np.zeros_like(binary_255, dtype=np.uint8)
	areas = stats[1:, cv2.CC_STAT_AREA]
	idx = int(np.argmax(areas)) + 1
	return (labels == idx).astype(np.uint8)


def _fill_holes(binary01: np.ndarray) -> np.ndarray:
	mask255 = (binary01 > 0).astype(np.uint8) * 255
	h, w = mask255.shape
	inv = (mask255 == 0).astype(np.uint8) * 255
	flood = inv.copy()
	ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
	cv2.floodFill(flood, ff_mask, (0, 0), 0)
	holes = flood == 255
	filled = mask255.copy()
	filled[holes] = 255
	return (filled > 0).astype(np.uint8)


def estimate_fundus_mask(image_rgb: np.ndarray, margin_px: int) -> np.ndarray:
	gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	white_ratio = float(th.mean() / 255.0)
	if white_ratio > 0.6:
		th = cv2.bitwise_not(th)
	cc = _largest_cc_mask(th)
	if int(cc.sum()) == 0:
		return np.ones((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
	k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
	cc255 = (cc * 255).astype(np.uint8)
	cc255 = cv2.morphologyEx(cc255, cv2.MORPH_CLOSE, k)
	cc = _fill_holes((cc255 > 0).astype(np.uint8))
	if margin_px > 0:
		k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
		cc = cv2.erode(cc.astype(np.uint8), k2)
	return (cc > 0).astype(np.uint8)


def component_elongation(coords_yx: np.ndarray) -> float:
	if coords_yx.shape[0] < 5:
		return 0.0
	xy = coords_yx[:, ::-1].astype(np.float32)
	xy -= xy.mean(axis=0, keepdims=True)
	cov = (xy.T @ xy) / max(1.0, float(xy.shape[0] - 1))
	try:
		w, _ = np.linalg.eigh(cov)
	except Exception:
		return 0.0
	w = np.sort(np.maximum(w, 1e-8))
	return float(w[-1] / w[0])


def zhang_suen_thinning(mask01: np.ndarray, max_iter: int = 50) -> np.ndarray:
	"""Zhang-Suen thinning. Input/output are {0,1}.

	This is slower than native ops but fine for 20 images at 512x512.
	"""
	img = (mask01 > 0).astype(np.uint8)
	changed = True
	iters = 0
	while changed and iters < max_iter:
		changed = False
		iters += 1
		for step in (0, 1):
			to_delete = []
			# avoid borders
			for i in range(1, img.shape[0] - 1):
				p2 = img[i - 1, 1:-1]
				p3 = img[i - 1, 2:]
				p4 = img[i, 2:]
				p5 = img[i + 1, 2:]
				p6 = img[i + 1, 1:-1]
				p7 = img[i + 1, :-2]
				p8 = img[i, :-2]
				p9 = img[i - 1, :-2]
				p1 = img[i, 1:-1]

				neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
				# count 0->1 transitions in ordered sequence p2,p3,...,p9,p2
				transitions = (
					((p2 == 0) & (p3 == 1)).astype(np.uint8)
					+ ((p3 == 0) & (p4 == 1)).astype(np.uint8)
					+ ((p4 == 0) & (p5 == 1)).astype(np.uint8)
					+ ((p5 == 0) & (p6 == 1)).astype(np.uint8)
					+ ((p6 == 0) & (p7 == 1)).astype(np.uint8)
					+ ((p7 == 0) & (p8 == 1)).astype(np.uint8)
					+ ((p8 == 0) & (p9 == 1)).astype(np.uint8)
					+ ((p9 == 0) & (p2 == 1)).astype(np.uint8)
				)

				cond0 = p1 == 1
				cond1 = (neighbors >= 2) & (neighbors <= 6)
				cond2 = transitions == 1
				if step == 0:
					cond3 = (p2 * p4 * p6) == 0
					cond4 = (p4 * p6 * p8) == 0
				else:
					cond3 = (p2 * p4 * p8) == 0
					cond4 = (p2 * p6 * p8) == 0

				del_mask = cond0 & cond1 & cond2 & cond3 & cond4
				if np.any(del_mask):
					xs = np.where(del_mask)[0] + 1
					to_delete.extend([(i, int(x)) for x in xs])

			if to_delete:
				for y, x in to_delete:
					img[y, x] = 0
				changed = True

	return img


def postprocess_mask(
	mask01: np.ndarray,
	image_rgb: np.ndarray,
	fundus_margin_px: int,
	frame_px: int,
	border_px: int,
	min_area_blob: int,
	min_area_line: int,
	elong_thr: float,
	thin: bool,
	dilate_px: int,
	cap_radius_px: int,
) -> np.ndarray:
	mask01 = (mask01 > 0).astype(np.uint8)
	if frame_px > 0:
		fp = int(frame_px)
		mask01[:fp, :] = 0
		mask01[-fp:, :] = 0
		mask01[:, :fp] = 0
		mask01[:, -fp:] = 0
	if int(mask01.sum()) == 0:
		return mask01

	fundus = estimate_fundus_mask(image_rgb, margin_px=fundus_margin_px)
	mask01 = (mask01 & fundus).astype(np.uint8)
	if int(mask01.sum()) == 0:
		return mask01

	fundus_full = estimate_fundus_mask(image_rgb, margin_px=0)
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
		coords = np.column_stack(np.where(labels == cid))
		elong = component_elongation(coords)
		near_border = bool(border_px > 0 and float(dist[labels == cid].min()) < float(border_px))

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

	# Optional: cap maximum vessel half-width while preserving thickness variation.
	# Idea: compute distance-transform radii, skeletonize, then reconstruct by dilating
	# skeleton pixels with radius=min(local_radius, cap_radius_px).
	if int(cap_radius_px) > 0 and int(keep.sum()) > 0:
		cap = int(cap_radius_px)
		dt = cv2.distanceTransform((keep > 0).astype(np.uint8), cv2.DIST_L2, 3)
		dt_i = np.clip(np.rint(dt).astype(np.int32), 0, cap)
		sk = zhang_suen_thinning(keep)
		out = np.zeros_like(keep, dtype=np.uint8)
		for r in range(1, cap + 1):
			seeds = ((sk > 0) & (dt_i == r)).astype(np.uint8)
			if int(seeds.sum()) == 0:
				continue
			k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
			out = np.maximum(out, cv2.dilate(seeds, k))
		keep = (out > 0).astype(np.uint8)
	elif thin and int(keep.sum()) > 0:
		# Legacy skeleton mode (will make vessels mostly uniform after dilation).
		sk = zhang_suen_thinning(keep)
		if dilate_px > 0:
			k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
			sk = cv2.dilate(sk.astype(np.uint8), k)
		keep = (sk > 0).astype(np.uint8)

	return keep


def run(args: argparse.Namespace) -> None:
	img_dir = Path(args.image_dir)
	mask_dir = Path(args.mask_dir)
	out_dir = Path(args.out_dir)
	ensure_dir(out_dir)

	images = {p.stem: p for p in list_images(img_dir)}
	masks = list_images(mask_dir)
	if not masks:
		raise FileNotFoundError(f"no masks found under: {mask_dir}")

	for mp in tqdm(masks, desc="postprocess"):
		stem = mp.stem
		if stem not in images:
			continue
		img = load_rgb(images[stem])
		img512 = np.array(Image.fromarray(img).resize((args.img_size, args.img_size), Image.Resampling.BILINEAR))
		m = np.array(Image.open(mp).convert("L"))
		m01 = (m > 0).astype(np.uint8)
		if m01.shape[0] != args.img_size or m01.shape[1] != args.img_size:
			m01 = cv2.resize(m01, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)

		pp = postprocess_mask(
			m01,
			img512,
			fundus_margin_px=args.fundus_margin_px,
			frame_px=args.frame_px,
			border_px=args.border_px,
			min_area_blob=args.min_area_blob,
			min_area_line=args.min_area_line,
			elong_thr=args.elong_thr,
			thin=args.thin,
			dilate_px=args.dilate_px,
			cap_radius_px=args.cap_radius_px,
		)
		out = Image.fromarray((pp > 0).astype(np.uint8) * 255, mode="L")
		out.save(out_dir / f"{stem}.png")

	print(f"saved: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Task5 postprocess (v2): remove outside/border noise, keep line-like vessels")
	p.add_argument("--image_dir", type=str, default="NeuMachineLearning-main/task5/segmentation/test/image")
	p.add_argument("--mask_dir", type=str, default="NeuMachineLearning-main/task5/segmentation/image")
	p.add_argument("--out_dir", type=str, default="NeuMachineLearning-main/task5/segmentation/image_pp")
	p.add_argument("--img_size", type=int, default=512)

	p.add_argument("--fundus_margin_px", type=int, default=12)
	p.add_argument("--frame_px", type=int, default=6)
	p.add_argument("--border_px", type=int, default=14)
	p.add_argument("--min_area_blob", type=int, default=120)
	p.add_argument("--min_area_line", type=int, default=25)
	p.add_argument("--elong_thr", type=float, default=10.0)

	p.add_argument("--thin", action="store_true")
	p.add_argument("--dilate_px", type=int, default=1)
	p.add_argument(
		"--cap_radius_px",
		type=int,
		default=0,
		help="Cap maximum vessel half-width (in pixels) while preserving thick/thin variation. Recommended instead of --thin.",
	)
	p.set_defaults(func=run)
	return p


def main() -> None:
	args = build_parser().parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
