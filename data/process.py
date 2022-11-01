import argparse
import os
import random
import pandas as pd
import numpy as np

import torchvision.transforms as transforms


class MyRotateTransform:
    def __init__(self, angles):#: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


def image_transforms(random_crop):
    if random_crop:
        full_transforms = transforms.Compose([
            transforms.RandomCrop((224, 224)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1, 0.05, 0, 0.1),
            MyRotateTransform(angles=[-90, 0, 90, 180]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        lim_transforms = transforms.Compose([
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        full_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1, 0.05, 0, 0.1),
            MyRotateTransform(angles=[-90, 0, 90, 180]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        lim_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return full_transforms, lim_transforms
    
    
def select_cohort(patch_labels, cohort):
    if cohort != 'all':
        patch_labels = patch_labels[patch_labels.cohort == cohort]
        patch_labels.reset_index(drop=True, inplace=True)
    print('Total number of slides:', len(patch_labels.slide.unique()))
    return patch_labels


def upsample(label_df, label_counts, predicting_var):
    upsample_factor = int(round(max(label_counts) / min(label_counts)))
    print('Upsample training set by a factor of', upsample_factor)
    uniq_slides = label_df.slide.unique()
    upsampled_slides = list(uniq_slides.copy())
    min_class = label_counts.argmin()
    min_labels = label_df[label_df[args.predicting_var]==min_class].reset_index()
    min_label_slides = list(min_labels.slide.unique())
    upsampled_df = label_df.copy()
    for i in range(upsample_factor-1):
        upsampled_slides += min_label_slides
        upsampled_df = upsampled_df.append(min_labels)
    print('-- Train Distribution after upsampling:')
    print(upsampled_df[args.predicting_var].value_counts())
    return upsampled_df.reset_index(drop=True), upsampled_slides  


def check_no_overlap(list1, list2):
    for item in list1:
        assert item not in list2
    for item in list2:
        assert item not in list1


def split_train_val(patch_labels, cohort, train_val_split, seed, prediction, predicting_var, upsample_data):
    patch_labels = select_cohort(patch_labels, cohort)
    
    # split train and val sets on case level
    cases = patch_labels.case.unique()
    num_train_cases = int(np.ceil(len(cases) * train_val_split))
    random.seed(seed)
    random.shuffle(cases)
    train_cases = cases[:num_train_cases]
    val_cases = cases[num_train_cases:]
    print('Number of train cases:', len(train_cases))
    print('Number of validation cases:', len(val_cases))
    
    check_no_overlap(train_cases, val_cases)
    
    train_patch_labels = patch_labels[patch_labels.case.isin(train_cases)].reset_index(drop=True)
    val_patch_labels = patch_labels[patch_labels.case.isin(val_cases)].reset_index(drop=True)
    
    if prediction in ['regression']:
        print('No upsampling')
        return train_patch_labels, val_patch_labels, val_cases, train_patch_labels.slide.unique()
    else:
        train_patch_counts = train_patch_labels[predicting_var].value_counts()
        if upsample_data:
            train_patch_labels, upsampled_slides = upsample(train_patch_labels, train_patch_counts, predicting_var)
        else:
            print('No upsampling')
        return train_patch_labels, val_patch_labels, val_cases, upsampled_train_slides
