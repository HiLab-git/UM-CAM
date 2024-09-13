import os
import cv2
import argparse
import numpy as np
import SimpleITK as sitk
from code_cls.utils.seed_points_statistics import *
from code_cls.utils.util import *


def fuse_EGDM_bf(D_b, D_f, ratio=4):
    P_EGD_f = np.exp(-D_f * ratio)
    P_EGD_b = 1 - np.exp(-D_b * ratio)

    P_EGD = np.where((P_EGD_b < 0.5), P_EGD_b, 0.5)
    P_EGD = np.where((P_EGD_f > 0.5), P_EGD_f, P_EGD)
    P_EGD = np.where(((P_EGD_b < 0.5) & (P_EGD_f > 0.5) & (D_b < D_f)), P_EGD_b, P_EGD)
    P_EGD = np.where(((P_EGD_b < 0.5) & (P_EGD_f > 0.5) & (D_f < D_b)), P_EGD_f, P_EGD)

    return P_EGD, P_EGD_b, P_EGD_f


def show_on_image(image, cam, save_name):
    img_norm = (image - image.min()) / (image.max() - image.min())
    img_norm = Image.fromarray(img_norm * 255.0)
    atten_map = cam * 255.0
    atten_map = Image.fromarray(pseudo_mask)
    img_scl = get_fused_heat_map(img_norm, atten_map)
    img_scl.save(save_name)


def show_cam_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    use_rgb: bool = False,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def sharpen(p, T):
    return p ** (1.0 / T) / (p ** (1.0 / T) + (1 - p) ** (1.0 / T))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate dataset for segmentation")
    parser.add_argument("--root_path", type=str, default="")
    parser.add_argument("--dir_cam", default="code_cls/result/umcam/")

    parser.add_argument("--stage", type=str, default="valid")
    parser.add_argument("--split", type=str, default="fold1")
    parser.add_argument("--fusion", type=str2bool, default=False)
    parser.add_argument("--fusion_method", type=str, default=None)
    parser.add_argument("--method", type=str, default="gradcam")
    
    parser.add_argument("--cam_thresh", type=float, default=0.40)
    parser.add_argument("--margin_x", type=int, default=5, help="margin for background seed points (width)")
    parser.add_argument("--margin_y", type=int, default=5, help="margin for background seed points (height)")
    parser.add_argument("--v", type=float, default=1e10, help="fast geodesic")
    parser.add_argument("--lamb", type=float, default=1.0, help="fast geodesic: lamb = 0.0 (Euclidean) or 1.0 (Geodesic) or (0.0, 1.0) (mixture)")
    parser.add_argument("--iterations", type=float, default=1, help="fast geodesic")
    parser.add_argument("--ratio", type=float, default=1, help="geodesic distance range ratio")

    args = parser.parse_args()

    args.root_path = f'{args.root_path}/{args.split}'
    dir_cam = f'{args.dir_cam}/{args.method}/{args.split}'
    
    Dice_all_metric, HD95_all_metric, Dice_norm_all_metric, HD95_norm_all_metric = [], [], [], []
    filenames = os.listdir(dir_cam)
    for filename in filenames:
        save_name = filename.split(".")[0]
        
        cam = np.load(f'{dir_cam}/{filename}')
        gt_map = sitk.GetArrayFromImage(sitk.ReadImage(f'{args.root_path}/label/{args.stage}/{save_name}.nii.gz')).squeeze()
        image = sitk.GetArrayFromImage(sitk.ReadImage(f'{args.root_path}/image/{args.stage}/{save_name}.nii.gz')).squeeze()
        assert cam.shape == gt_map.shape
        
        min_intra = 1e6
        best_thresh = 0.0
        for i in range(1, 20):
            thresh = 0.05 * i

            """Minimizing intra-class invariance"""
            mask_fg = np.where(((cam >= thresh)), 1, 0)
            mask_bg = np.where(((cam < thresh)), 1, 0)

            intensity_mean_bg = np.sum(mask_bg * image) / np.sum(mask_bg + 1e-6)
            intensity_mean_fg = np.sum(mask_fg * image) / np.sum(mask_fg + 1e-6)
            intra_bg = np.sum((image - intensity_mean_bg) ** 2 * mask_bg) / np.sum(mask_bg + 1e-6)
            intra_fg = np.sum((image - intensity_mean_fg) ** 2 * mask_fg) / np.sum(mask_fg + 1e-6)
            intra = (intra_bg + intra_fg) / 2
            if intra < min_intra:
                min_intra = intra
                best_thresh = thresh
        
        pseudo_mask = np.zeros_like(cam)
        pseudo_mask[cam >= best_thresh] = 1
        pseudo_mask = largestConnectComponent(pseudo_mask)
        
        margin_x, margin_y = args.margin_x, args.margin_y
        if pseudo_mask.sum():
            """brats"""
            _, (cx, cy), _ = get_seed_points(pseudo_mask, margin_x, margin_y)

            if image[cx, cy] == 0:
                pseudo_label = np.zeros_like(cam)
            else:
                pseudo_mask_bg = np.zeros_like(cam)
                pseudo_mask_bg[np.where(cam > best_thresh)] = 1
                pseudo_mask_bg = largestConnectComponent(pseudo_mask_bg)
                _, _, (x, y, w, h) = get_seed_points(pseudo_mask_bg, margin_x, margin_y)

                # show_on_image(image, pseudo_mask, f'{basepath_EGDM_fg_png}/{save_name}.png')
                # show_on_image(image, pseudo_mask_bg, f'{basepath_EGDM_bg_png}/{save_name}.png')

                if (cx < x) or (cx > x + w) or (cy < y) or (cy > y + h):
                    cx = int(x + w / 2)
                    cy = int(y + h / 2)
                
                """visualize the seed points"""
                # seeds_size = 2
                # img_seeds = ((image - image.min()) / (image.max() - image.min()) * 255.0)
                # img_seeds = np.array([img_seeds, img_seeds, img_seeds], dtype=np.uint8)
                # img_seeds = np.transpose(img_seeds, [1, 2, 0])
                # img_seeds = np.ascontiguousarray(img_seeds)
                # cv2.circle(img_seeds, (cy, cx), seeds_size, (0, 0, 255), -1)
                # cv2.circle(img_seeds, (y, x), seeds_size, (0, 255, 0), -1)
                # cv2.circle(img_seeds, (y, x + w), seeds_size, (0, 255, 0), -1)
                # cv2.circle(img_seeds, (y + h, x), seeds_size, (0, 255, 0), -1)
                # cv2.circle(img_seeds, (y + h, x + w), seeds_size, (0, 255, 0), -1)
                # cv2.imwrite(os.path.join(basepath_seeds, save_name + ".png"), img_seeds)


                """foreground (target)"""
                img = (image - image.mean()) / image.std()
                img = np.asanyarray(img, np.float32)

                seed_points_fg = [(cx, cy)]
                geodesic_dist_fg = geodis_fast(img, seed_points_fg, v=args.v, lamb=args.lamb, iterations=args.iterations)


                """background (target)"""
                seed_points_bg = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
                geodesic_dist_bg = geodis_fast(img, seed_points_bg, v=args.v, lamb=args.lamb, iterations=args.iterations)
                geodesic_dist = np.array([geodesic_dist_bg, geodesic_dist_fg])
                geodesic_dist = geodesic_dist.min(0)
                
                weight_map = np.exp(-geodesic_dist * args.ratio)
                weight_map = (weight_map - 0.5) / 0.5
                weight_map[weight_map < 0] = 0

                P_EGD, P_EGD_b, P_EGD_f = fuse_EGDM_bf(geodesic_dist_bg, geodesic_dist_fg, ratio=args.ratio)
                pseudo_label = weight_map * P_EGD + (1 - weight_map) * cam
                pseudo_label = max_min_norm(pseudo_label)
