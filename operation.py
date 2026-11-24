"""
operation.py – K-Means clustering และการจัดกลุ่มรูปภาพ

การเปลี่ยนแปลงจากต้นฉบับ:
- แก้ typo: Custering → Clustering, group_custering → group_clustering
- ลบ print(f"Change: {change}") ที่เป็น debug print → เพิ่ม verbose=False แทน
- เพิ่ม random_state สำหรับ reproducibility (ผลลัพธ์เหมือนกันทุกครั้งเมื่อกำหนด seed)
- เพิ่ม max_iter=300 ป้องกัน infinite loop กรณี convergence ช้ามาก
- ลบ import cv2 ที่ซ้ำ (ใช้อยู่แล้วใน group_clustering)
"""
import os
import shutil

import cv2
import numpy as np


class Clustering:
    """K-Means clustering สำหรับ pixel vector ของรูปภาพ

    Parameters
    ----------
    random_state : int หรือ None
        seed สำหรับการสุ่ม initial centroids
        None = สุ่มใหม่ทุกครั้ง (ผลลัพธ์อาจต่างกันแต่ละ run)
    """

    def __init__(self, random_state: int | None = 42) -> None:
        self.random_state = random_state

    def kmeans(
        self,
        x: np.ndarray,
        k: int,
        mode: str = "manhattan",
        max_iter: int = 300,
        verbose: bool = False,
    ) -> np.ndarray:
        """รัน K-Means แล้วคืน cluster index ของแต่ละตัวอย่าง

        Parameters
        ----------
        x        : (n_samples, n_features) pixel vectors
        k        : จำนวน cluster
        mode     : 'manhattan' หรือ 'square_euclidean'
        max_iter : จำนวนรอบสูงสุด (ป้องกัน infinite loop)
        verbose  : True = แสดง centroid change ทุก iteration

        Returns
        -------
        idx : np.ndarray, shape (n_samples,)  – cluster index
        """
        rng = np.random.default_rng(self.random_state)
        x = x.astype(np.float64)
        centroids = x[rng.permutation(len(x))[:k]]

        for iteration in range(max_iter):
            # คำนวณ distance จากทุก sample ไปทุก centroid
            D = np.zeros((len(x), k))
            for i in range(k):
                if mode == "manhattan":
                    D[:, i] = np.mean(np.abs(x - centroids[i]), axis=1)
                elif mode == "square_euclidean":
                    D[:, i] = np.mean((x - centroids[i]) ** 2, axis=1)
                else:
                    raise ValueError(f"mode ไม่รู้จัก: {mode!r}  (ใช้ 'manhattan' หรือ 'square_euclidean')")

            idx = np.argmin(D, axis=1)

            # อัพเดต centroid (cluster ว่างคงเดิม)
            c_old = centroids.copy()
            for i in range(k):
                if np.any(idx == i):
                    centroids[i] = x[idx == i].mean(axis=0)

            change = np.mean(np.abs(c_old - centroids))
            if verbose:
                print(f"  iter {iteration + 1:3d}  change={change:.6f}")

            if change < 1e-6:
                break

        return idx

    def group_clustering(
        self,
        file_names: list[str],
        labels: np.ndarray,
        img_source_folder: str,
        cluster_output_path: str,
    ) -> None:
        """คัดลอกรูปภาพเข้า folder ตาม cluster

        Parameters
        ----------
        file_names          : ชื่อไฟล์ตามลำดับเดียวกับ labels
        labels              : cluster index ของแต่ละรูป
        img_source_folder   : folder ต้นทางที่มีรูปภาพ
        cluster_output_path : folder ปลายทาง (ถูกลบและสร้างใหม่ทุกครั้ง)
        """
        if os.path.abspath(cluster_output_path) == os.path.abspath(img_source_folder):
            raise ValueError(
                "Output folder ต้องไม่ใช่ folder เดียวกับ source: "
                "Output จะถูกลบทุกครั้งที่รัน"
            )
        shutil.rmtree(cluster_output_path, ignore_errors=True)
        os.makedirs(cluster_output_path)

        for name, label in zip(file_names, labels):
            cluster_folder = os.path.join(cluster_output_path, f"cluster_{label}")
            os.makedirs(cluster_folder, exist_ok=True)
            img = cv2.imread(os.path.join(img_source_folder, name))
            if img is None:
                continue
            cv2.imwrite(os.path.join(cluster_folder, name), img)
