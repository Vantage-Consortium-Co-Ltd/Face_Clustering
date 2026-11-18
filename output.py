import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np

from data import IMG_SIZE


def save_clusters(names, labels, src, dst):
    """ก๊อปรูปเดิมไปไว้ใน dst/cluster_<n>/ (ลบ dst ทิ้งก่อนทุกครั้ง จะได้เหลือแค่ผลรอบล่าสุด)"""
    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError('โฟลเดอร์ output ต้องไม่ใช่โฟลเดอร์ dataset เพราะจะถูกลบทุกครั้ง')
    shutil.rmtree(dst, ignore_errors=True)
    for name, label in zip(names, labels):
        folder = os.path.join(dst, f'cluster_{label}')
        os.makedirs(folder, exist_ok=True)
        shutil.copy2(os.path.join(src, name), folder)       # ก๊อปไฟล์ตรง ๆ ไม่ต้อง encode ใหม่


def plot_clusters(X, names, labels, centroids, path, per_row=8):
    """รูปเดียวรวมทุก cluster: แถวละ 1 cluster ช่องแรกคือ centroid (หน้าเฉลี่ย) ตามด้วยตัวอย่างสมาชิก"""
    w, h = IMG_SIZE
    k = len(centroids)
    fig, axes = plt.subplots(k, per_row + 1, figsize=((per_row + 1) * 1.4, k * 2), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for i in range(k):
        members = np.flatnonzero(labels == i)
        axes[i, 0].imshow(centroids[i].reshape(h, w), cmap='gray')
        axes[i, 0].set_title(f'cluster {i}\n({len(members)} imgs)', fontsize=8, fontweight='bold')
        for ax, j in zip(axes[i, 1:], members[:per_row]):
            ax.imshow(X[j].reshape(h, w), cmap='gray')
            ax.set_title(os.path.splitext(names[j])[0][:12], fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig
