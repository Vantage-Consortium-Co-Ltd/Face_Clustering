import os
import numpy as np
import cv2
class Custering :
    def kmeans_calculate(self, x, k, mode='manhattan'):
        x = x.astype(np.float64)
        centroids = x[np.random.permutation(len(x))[:k]]
        while True:
            D = np.zeros((len(x), k))
            for i in range(k):
                if mode == 'manhattan':
                    D[:, i] = np.mean(np.abs(x - centroids[i]), axis=1)
                elif mode == 'euclidean':
                    D[:, i] = np.mean((x - centroids[i])**2, axis=1)

            idx = np.argmin(D, axis=1)
            c_old = centroids.copy()
            for i in range(k):
                if np.any(idx == i):
                    centroids[i] = np.mean(x[idx == i], axis=0)

            change = np.mean(np.abs(c_old - centroids))
            print(f"Change: {change}")
            if change < 1e-6:
                break
    
        return idx

    def group_custering(self, file_names, labels, img_source_folder, cluster_output_path):
        os.makedirs(cluster_output_path, exist_ok=True)
        for name, label in zip(file_names, labels):
            cluster_folder = os.path.join(cluster_output_path, f"cluster_{label}")
            os.makedirs(cluster_folder, exist_ok=True)
            img = cv2.imread(os.path.join(img_source_folder, name))
            if img is None:
                continue
            cv2.imwrite(os.path.join(cluster_folder, name), img)

