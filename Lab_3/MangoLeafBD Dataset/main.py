import os
import cv2
import albumentations as A
from albumentations.core.composition import OneOf
# from albumentations.augmentations.transforms import (
#     HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast, ShiftScaleRotate
# )
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast, ShiftScaleRotate

# Define augmentation pipeline
augmentation_pipeline = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
])

# Paths
input_folder = "Lab_3/MangoLeafBD Dataset"
output_folder = "Lab_3/MangoLeafBD_Augmented"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Apply augmentations
for category in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category)
    augmented_category_path = os.path.join(output_folder, category)
    os.makedirs(augmented_category_path, exist_ok=True)

    if os.path.isdir(category_path):
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            image = cv2.imread(image_path)

            if image is not None:
                augmented = augmentation_pipeline(image=image)
                augmented_image = augmented["image"]

                # Save augmented image
                augmented_image_path = os.path.join(augmented_category_path, f"aug_{image_name}")
                cv2.imwrite(augmented_image_path, augmented_image)

print("Image augmentation completed.")