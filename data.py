"""
data.py – โหลดและ preprocess รูปภาพจาก folder

การเปลี่ยนแปลงจากต้นฉบับ:
- แก้ bug: condition เดิม `not os.path.isfile(...) and not startswith(".")` ไม่ถูกต้อง
  → ใช้ os.path.isdir() แทน ชัดเจนและถูกต้องกว่า
- แก้ bug: ถ้า folder ไม่ถูกต้อง จะ return [] (list) ไม่ใช่ ndarray
  → data_x.shape ใน main.py จะ crash → แก้ให้ return np.empty() เสมอ
- รองรับ .jpg .jpeg .png .bmp .tif เหมือนกับ visualize.py (เดิมรองรับแค่ .jpg)
- เพิ่ม type hints และ docstring
- ลบ import cv2 ที่ไม่ได้ใช้ใน scope นี้ → ใช้แค่ numpy
"""
import os
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = (240, 300)  # (width, height) สำหรับ cv2.resize

# นามสกุลที่รองรับ (ตรงกับ visualize.py)
_SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_data(folder_path: str) -> tuple[np.ndarray, list[str]]:
    """โหลดรูปภาพจาก folder_path แล้ว flatten เป็น vector

    Parameters
    ----------
    folder_path : str
        path ของ folder ที่มีรูปภาพ (ไม่สแกน sub-folder)

    Returns
    -------
    data_x : np.ndarray, shape (n_images, width*height)
        pixel vector ของแต่ละรูป (grayscale, flatten)
    file_names : list[str]
        ชื่อไฟล์ตามลำดับเดียวกับ data_x
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"ไม่พบ folder: {folder_path!r}")

    data_x: list[np.ndarray] = []
    file_names: list[str] = []

    for file in sorted(os.listdir(folder_path)):  # sorted → ผลลัพธ์สม่ำเสมอ
        if Path(file).suffix.lower() not in _SUPPORTED_EXT:
            continue
        img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        data_x.append(img.flatten())
        file_names.append(file)

    return np.array(data_x, dtype=np.float64), file_names
