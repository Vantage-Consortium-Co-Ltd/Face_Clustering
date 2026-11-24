"""
visualize.py – แสดงรูปภาพแต่ละ cluster ในหน้าต่าง matplotlib

การเปลี่ยนแปลงจากต้นฉบับ:
- ลบ class Visualization ที่มีแค่ method เดียวออก → ใช้ฟังก์ชัน visualize() ตรง ๆ
  (class ที่มีแค่ method เดียวและไม่มี state เป็น overhead ที่ไม่จำเป็น)
- ลบ window-positioning try/except block ออก (อ่านยาก, ใช้ประโยชน์น้อย,
  backend ส่วนใหญ่ไม่รองรับ)
- เพิ่ม type hints
"""
import os

import matplotlib.pyplot as plt


_SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def visualize(folder_output: str, cols: int = 5) -> None:
    """แสดงหน้าต่าง matplotlib หนึ่งหน้าต่างต่อ cluster

    Parameters
    ----------
    folder_output : str
        path ของ folder ที่มี sub-folder cluster_0, cluster_1, ...
    cols : int
        จำนวนคอลัมน์ต่อแถวในแต่ละหน้าต่าง (default 5)
    """
    for sub_folder in sorted(os.listdir(folder_output)):
        folder = os.path.join(folder_output, sub_folder)
        if not os.path.isdir(folder):
            continue

        files = sorted(
            f for f in os.listdir(folder) if f.lower().endswith(_SUPPORTED_EXT)
        )
        if not files:
            continue

        rows = -(-len(files) // cols)  # ceil division
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * 2.2, rows * 2.6 + 0.6),
            squeeze=False,
        )
        fig.suptitle(
            f"{sub_folder.replace('_', ' ').title()}  ({len(files)} images)",
            fontweight="bold",
        )
        for ax in axes.ravel():
            ax.axis("off")
        for ax, f in zip(axes.ravel(), files):
            ax.imshow(plt.imread(os.path.join(folder, f)))
            ax.set_title(os.path.splitext(f)[0], fontsize=8)

        fig.tight_layout()

    plt.show()
