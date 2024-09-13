import glob
import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk

sys.path.append(os.getcwd())
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])


def get_umcam(cams):
    uncertainty = -(cams * np.log(cams + 1e-6) + (1 - cams) * np.log((1 - cams) + 1e-6))
    weight = np.exp(-uncertainty)

    cam_mr = np.sum(cams * weight, 0) / (np.sum(weight, 0) + 1e-6)
    cam_mr = (cam_mr - cam_mr.min()) / (cam_mr.max() - cam_mr.min())

    return cam_mr, uncertainty


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cam fusion")
    parser.add_argument("--root_path", type=str, default="")
    parser.add_argument("--dir_cam", default="code_cls/result/cam")
    parser.add_argument("--save_dir", type=str, default="code_cls/result/umcam")
    parser.add_argument("--stage", type=str, default="valid")
    parser.add_argument("--split", type=str, default="fold1")
    parser.add_argument("--layer_num", nargs="+", help="<Required> Set flag", required=True)
    parser.add_argument("--fusion_method", type=str, default="umcam")
    parser.add_argument("--checkpoint", type=str, default="best.pth",)
    parser.add_argument( "--method", type=str, default="gradcam", help="Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam/eigencam/eigengradcam/layercam",)

    sys.path.append(os.getcwd())
    args = parser.parse_args()
    args.root_path = "{}/{}/image".format(args.root_path, args.split)

    args.save_dir = f'{args.save_dir}/{args.method}/{args.split}'
    os.makedirs(args.save_dir)

    """load data (load the file according to the folder)"""
    filenames = glob.glob(args.root_path + "/" + args.stage + "/*.nii.gz")    
    for filename in filenames:
        person_name = os.path.basename(filename).split(".")[0]
        print(person_name)
        
        """load cams"""
        cams = []
        for layer_num in args.layer_num:
            cam_dir = os.path.join(args.dir_cam, "layer_" + layer_num, args.method, args.stage, "npy")
            cam_name = "{}/{}".format(cam_dir, person_name + ".npy")
            if os.path.exists(cam_name):
                cam = np.load(cam_name)
            else:
                cam = sitk.GetArrayFromImage(sitk.ReadImage(filename)).squeeze()
            cams.append(cam)
        cams = np.array(cams)

        """generate umcam"""
        cam_mr, uncertainty = get_umcam(cams)
        np.save("{}/{}".format(args.save_dir, person_name), cam_mr)
            
