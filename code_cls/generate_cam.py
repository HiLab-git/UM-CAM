from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.util import *
from network.vgg import vgg16, vgg16_DRS
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
    FullGrad,
)
from utils.dataset import *
import argparse
import cv2
import sys
import os
import numpy as np
import torch
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.getcwd())
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true", default=True, help="Use NVIDIA GPU acceleration")
    parser.add_argument("--root_path", type=str, default="")
    parser.add_argument("--dir_cam", default="code_cls/result/cam/")

    parser.add_argument("--dataset_name", type=str, default="fetal_brain", choices=["fetal_brain", "brats2020"])  # need revision
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stage", type=str, default="valid")
    parser.add_argument("--split", type=str, default="fold1")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--aug_smooth", action="store_true", help="Apply test time augmentation to smooth the CAM")
    parser.add_argument("--eigen_smooth", action="store_true", help="Reduce noise by taking the first principle componenet of cam_weights*activations")
    parser.add_argument("--method", type=str, default="gradcam", choices=["gradcam", "gradcam++", "scorecam", "xgradcam", "ablationcam", "eigencam", "eigengradcam", "layercam", "fullgrad"])
    parser.add_argument("--checkpoint", type=str, default="best.pth")
    parser.add_argument("--model_name", type=str, default="vgg16", choices=["vgg16", "vgg16_DRS"])
    parser.add_argument("--layer_num", type=str, default="30")
    parser.add_argument("--delta", type=float, default=0.55, help="set 0 for the learnable DRS")

    args = parser.parse_args()
    args.root_path = "{}/{}/image".format(args.root_path, args.split)

    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def get_model(args):
    """load pretrained model"""
    if args.model_name == "vgg16":
        model = vgg16(pretrained=True)
    elif args.model_name == "vgg16_DRS":
        model = vgg16_DRS(pretrained=True, delta=args.delta)
    
    model = model.cuda()
    model.eval()
    ckpt = torch.load(args.checkpoint, map_location="cuda:0")
    model.load_state_dict(ckpt["model"], strict=True)

    return model


def get_target_layer(args):
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    if int(args.layer_num) < 30:
        target_layer = [model.features[int(args.layer_num)]]
    elif int(args.layer_num) == 30:
        if args.model_name == "vgg16":
            target_layer = [model.extra_conv]
    else:
        print("Please check the target layer number")
    return target_layer


if __name__ == "__main__":
    """python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    output_dir = f'{args.dir_cam}/layer_{args.layer_num}/{args.method}/{args.stage}'
    output_dir_cam = f'{output_dir}/cam'
    output_dir_graycam_npy = f'{output_dir}/npy'
    
    os.makedirs(output_dir_cam, exist_ok=True)
    os.makedirs(output_dir_graycam_npy, exist_ok=True)

    pred_label_csv_name = os.path.join(output_dir, "pred_label_slice_" + args.stage + ".csv")

    """load data"""
    data_loader = test_dataloader(args, split=args.stage)

    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
    }

    model = get_model(args)
    model.eval()
    target_layers = get_target_layer(args)

    pred_label_csv, cls_pred_all, cls_label_all = [], [], []
    for idx, data in enumerate(data_loader):
        print("[%04d/%04d]" % (idx, len(data_loader)), end="\r")
        img, label, img_name = data["image"], data["label"], data["filename"]
        raw_img = sitk.GetArrayFromImage(sitk.ReadImage(f'{args.root_path}/{args.stage}/{img_name[0]}')).squeeze()
        
        img.requires_grad_()
        H, W = raw_img.shape
        rgb_img = img.squeeze()
        rgb_img = rgb_img.detach().cpu().numpy()
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
        rgb_img = np.float32() / 255

        input_tensor = img
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        target_category = None

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = methods[args.method]
        with cam_algorithm(
            model=model, target_layers=target_layers, use_cuda=args.use_cuda
        ) as cam:
            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 1
            grayscale_cam, target_category = cam(
                input_tensor=input_tensor,
                target_category=target_category,
                aug_smooth=args.aug_smooth,
                eigen_smooth=args.eigen_smooth,
            )  # target_category

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0]
            grayscale_cam = cv2.resize(grayscale_cam, [W, H])

        cls_pred_all.append(target_category[0])
        cls_label_all.append(label.data.item())

        save_name = os.path.basename(img_name[0]).split(".")[0]
        if target_category:
            img_norm = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min())
            img_norm = Image.fromarray(img_norm * 255.0)
            # atten_map = grayscale_cam * 255.0
            atten_map = Image.fromarray(grayscale_cam)
            img_scl = get_fused_heat_map(img_norm, atten_map)

            img_scl.save(f"{output_dir_cam}/{save_name}.jpg")
            np.save(f"{output_dir_graycam_npy}/{save_name}", grayscale_cam)
            
        pred_label_csv.append({"filename": save_name, "label": label.data.item(), "pred_label": target_category[0]})

    dataframe = pd.DataFrame(pred_label_csv, columns=["filename", "label", "pred_label"])
    dataframe.to_csv(pred_label_csv_name, index=False)

    """compute clssification results"""
    val_cls_acc = accuracy_score(cls_label_all, cls_pred_all)
    val_cls_prec = precision_score(cls_label_all, cls_pred_all, zero_division=0)
    val_cls_recall = recall_score(cls_label_all, cls_pred_all, zero_division=0)
    val_cls_f1 = f1_score(cls_label_all, cls_pred_all, zero_division=0)
    print("classification accuracy: %.4f, precision: %.4f, recall: %.4f, f1: %.4f" % (val_cls_acc, val_cls_prec, val_cls_recall, val_cls_f1))
    
