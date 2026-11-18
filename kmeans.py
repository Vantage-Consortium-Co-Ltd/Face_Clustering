import numpy as np


class KMeans:
    """K-means เขียนเองด้วย NumPy

    distance = 'euclidean' -> จุดศูนย์กลาง = ค่าเฉลี่ย (mean) ของสมาชิก  (K-means ปกติ)
    distance = 'manhattan' -> จุดศูนย์กลาง = ค่ามัธยฐาน (median) ของสมาชิก (K-medians)
    ใช้ median คู่กับ manhattan เพราะ median คือจุดที่ทำให้ผลรวมระยะ manhattan ต่ำสุด"""

    def __init__(self, k=3, distance='euclidean', n_init=10, max_iter=100, seed=0):
        self.k = k
        self.distance = distance
        self.n_init = n_init        # สุ่มจุดเริ่มกี่รอบ แล้วเก็บรอบที่ดีที่สุด
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

    def _dist(self, X, C):
        """ระยะจากทุกจุดไปทุก centroid -> (จำนวนจุด, k)"""
        if self.distance == 'manhattan':
            return np.stack([np.abs(X - c).sum(axis=1) for c in C], axis=1)
        return np.stack([((X - c) ** 2).sum(axis=1) for c in C], axis=1)

    def _center(self, points):
        return np.median(points, axis=0) if self.distance == 'manhattan' else points.mean(axis=0)

    def _init_plus_plus(self, X):
        """k-means++: จุดแรกสุ่ม จุดถัดไปสุ่มโดยให้จุดที่ไกลจาก centroid เดิมมีโอกาสถูกเลือกมากกว่า
        ช่วยไม่ให้ centroid เริ่มต้นกระจุกกัน"""
        C = [X[self.rng.integers(len(X))]]
        for _ in range(1, self.k):
            d = self._dist(X, C).min(axis=1)
            if d.sum() == 0:            # ทุกจุดทับ centroid เดิมหมด (รูปซ้ำกัน) -> สุ่มแบบเท่ากันแทน กันหารศูนย์
                C.append(X[self.rng.integers(len(X))])
            else:
                C.append(X[self.rng.choice(len(X), p=d / d.sum())])
        return np.array(C)

    def _run_once(self, X):
        C = self._init_plus_plus(X)
        for it in range(self.max_iter):
            labels = self._dist(X, C).argmin(axis=1)
            new_C = np.array([self._center(X[labels == i]) if np.any(labels == i) else C[i]
                              for i in range(self.k)])      # cluster ว่าง -> ใช้ centroid เดิม
            if np.allclose(new_C, C):
                break
            C = new_C
        D = self._dist(X, C)
        labels = D.argmin(axis=1)                             # กลุ่มตาม centroid สุดท้าย ให้ตรงกับ C ที่คืนไป
        cost = D.min(axis=1).sum()                            # ผลรวมระยะของทุกจุดถึง centroid ของตัวเอง
        return labels, C, cost, it + 1

    def fit(self, X):
        best = None
        for r in range(self.n_init):
            labels, C, cost, iters = self._run_once(X)
            print(f'run {r + 1:2d}: {iters:3d} iterations, cost = {cost:.2f}')
            if best is None or cost < best[2]:
                best = (labels, C, cost)
        self.labels_, self.centroids_, self.cost_ = best
        return self
