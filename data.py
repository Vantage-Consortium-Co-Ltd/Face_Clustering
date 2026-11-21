import os

import cv2
import numpy as np

IMG_SIZE = (140, 180)   # (กว้าง, สูง) ทุกรูปต้องขนาดเท่ากัน ไม่งั้นเวกเตอร์ยาวไม่เท่ากัน


def load_images(folder):
    """อ่านทุก .jpg ใน folder (ไม่เข้าโฟลเดอร์ย่อย) -> X (1 แถว = 1 รูป), ชื่อไฟล์เรียงตาม X"""
    X, names = [], []
    for f in sorted(os.listdir(folder)):
        # if not f.lower().endswith(.png):
        #    png_convert(f)
        if not f.lower().endswith('.jpg'):
            continue
        # ใช้ imdecode แทน cv2.imread เพราะ cv2.imread อ่าน path ที่มีภาษาไทยบน Windows ไม่ได้ (ได้ None)
        img = cv2.imdecode(np.fromfile(os.path.join(folder, f), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        X.append(cv2.resize(img, IMG_SIZE).flatten())
        names.append(f)
    # หาร 255 ให้ค่าอยู่ช่วง 0-1 ระยะจะได้ไม่ใหญ่เกิน
    return np.array(X, dtype=np.float64) / 255, names
