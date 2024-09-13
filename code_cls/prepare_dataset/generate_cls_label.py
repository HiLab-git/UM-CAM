# generate classification label

import os
import pandas as pd
import SimpleITK as sitk


def generate_cls_label(label):
    if label.sum():
        cls_label = 1
    else:
        cls_label = 0

    return cls_label


if __name__ == "__main__":
    dir = '/home/data/FJ/Dataset/Fetal_brain/Fetal_weak/slice_noise/label/train'
    cases = os.listdir(dir)
    
    csv_cls_label = []
    for case in cases:
        label = sitk.GetArrayFromImage(sitk.ReadImage(f'{dir}/{case}')).squeeze()
        cls_label = generate_cls_label(label)

        csv_cls_label.append({'filename': case.split('.')[0], 'label': cls_label})

    csv_name = 'cls_label.csv'
    dataframe = pd.DataFrame(csv_cls_label, columns=['filename', 'label'])
    dataframe.to_csv(csv_name, index=False)
