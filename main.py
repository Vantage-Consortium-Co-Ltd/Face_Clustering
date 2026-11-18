"""จัดกลุ่มรูปหน้าด้วย K-means: python main.py [--k 3] [--distance euclidean|manhattan] [--data Dataset]"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from data import load_images
from kmeans import KMeans
from output import save_clusters, plot_clusters

OUTPUT_DIR = 'Output'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=3, help='จำนวนกลุ่ม')
    ap.add_argument('--distance', default='euclidean', choices=['euclidean', 'manhattan'])
    ap.add_argument('--data', default='Dataset', help='โฟลเดอร์รูป (.jpg วางรวมกัน ไม่แยกโฟลเดอร์ย่อย)')
    ap.add_argument('--no-show', action='store_true', help='บันทึกรูปอย่างเดียว ไม่เปิดหน้าต่าง')
    args = ap.parse_args()

    X, names = load_images(args.data)
    if len(X) < args.k:
        raise SystemExit(f'มีรูป {len(X)} รูป น้อยกว่าจำนวนกลุ่ม k={args.k}')
    print('Data shape:', X.shape)

    model = KMeans(k=args.k, distance=args.distance).fit(X)
    print(f'\nbest cost = {model.cost_:.2f}')
    for name, label in zip(names, model.labels_):
        print(f'{name:<30} cluster {label}')
    print('ขนาดแต่ละกลุ่ม:', np.bincount(model.labels_, minlength=args.k).tolist())

    save_clusters(names, model.labels_, args.data, OUTPUT_DIR)
    plot_clusters(X, names, model.labels_, model.centroids_, os.path.join(OUTPUT_DIR, 'clusters.png'))
    print(f'\nบันทึกไว้ที่ {OUTPUT_DIR}/cluster_<n>/ และ {OUTPUT_DIR}/clusters.png')
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
