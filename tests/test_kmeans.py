"""เทสต์ KMeans (euclidean / manhattan) และการอ่านรูป ด้วยข้อมูลสังเคราะห์ (ไม่ใช้รูปหน้าจริง)
รัน: pytest"""
import cv2
import numpy as np
import pytest

from data import IMG_SIZE, load_images
from kmeans import KMeans


def three_blobs(n=20, seed=0):
    """3 กลุ่มที่อยู่ห่างกันชัดเจน + label จริง"""
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    X = np.vstack([c + rng.normal(0, 0.5, (n, 2)) for c in centers])
    return X, np.repeat(np.arange(3), n)


def same_grouping(a, b):
    """เลข cluster สลับกันได้ (0 ของเราอาจเป็น 2 ของอีกฝั่ง) จึงเทียบว่าแบ่งกลุ่มเหมือนกันไหม"""
    return all(len(set(a[b == g])) == 1 for g in np.unique(b)) and len(set(a)) == len(set(b))


@pytest.mark.parametrize('distance', ['euclidean', 'manhattan'])
def test_แยก_3_กลุ่มที่ห่างกันได้ถูก(distance, capsys):
    X, truth = three_blobs()
    model = KMeans(k=3, distance=distance, n_init=3).fit(X)
    assert same_grouping(model.labels_, truth)
    assert model.centroids_.shape == (3, 2)


def test_centroid_ของ_manhattan_คือมัธยฐาน_ไม่โดนค่าโด่งดึง(capsys):
    # กลุ่มเดียว มีค่าโด่ง 1 ตัว: mean ถูกดึงไปไกล แต่ median ไม่ขยับ
    X = np.array([[0.], [1.], [2.], [3.], [100.]])
    mean_c = KMeans(k=1, distance='euclidean', n_init=1).fit(X).centroids_[0, 0]
    med_c = KMeans(k=1, distance='manhattan', n_init=1).fit(X).centroids_[0, 0]
    assert mean_c == pytest.approx(21.2)
    assert med_c == pytest.approx(2.0)


def test_labels_ตรงกับ_centroid_สุดท้ายแม้หยุดก่อนนิ่ง(capsys):
    X, _ = three_blobs(seed=3)
    model = KMeans(k=3, max_iter=1, n_init=1, seed=5).fit(X)
    nearest = model._dist(X, model.centroids_).argmin(axis=1)
    assert np.array_equal(model.labels_, nearest)


def test_cost_ต่ำสุดจากหลายรอบ(capsys):
    X, _ = three_blobs()
    many = KMeans(k=3, n_init=10).fit(X).cost_
    one = KMeans(k=3, n_init=1).fit(X).cost_
    assert many <= one + 1e-9


def test_รูปซ้ำกันหมดไม่พัง(capsys):
    # k-means++ เดิมหาร d.sum() = 0 แล้วได้ NaN -> ต้องไม่ error
    X = np.ones((6, 4))
    model = KMeans(k=2, n_init=2).fit(X)
    assert model.cost_ == 0
    assert not np.isnan(model.centroids_).any()


def test_ผลเหมือนเดิมเมื่อ_seed_เท่ากัน(capsys):
    X, _ = three_blobs()
    a = KMeans(k=3, seed=7).fit(X)
    b = KMeans(k=3, seed=7).fit(X)
    assert np.array_equal(a.labels_, b.labels_)


# ---------- load_images ----------
def save(path, img):
    cv2.imencode('.jpg', img)[1].tofile(str(path))


def test_load_images_ย่อขนาดและสเกล_0_ถึง_1(tmp_path):
    save(tmp_path / 'b.jpg', np.full((50, 30), 255, np.uint8))
    save(tmp_path / 'a.JPG', np.zeros((300, 200), np.uint8))       # นามสกุลตัวใหญ่ก็ต้องอ่าน
    (tmp_path / 'note.txt').write_text('skip', encoding='utf-8')
    X, names = load_images(str(tmp_path))
    assert names == ['a.JPG', 'b.jpg']
    assert X.shape == (2, IMG_SIZE[0] * IMG_SIZE[1])
    assert X.min() >= 0 and X.max() <= 1


def test_load_images_โฟลเดอร์ภาษาไทยก็อ่านได้(tmp_path):
    folder = tmp_path / 'รูปหน้า'
    folder.mkdir()
    save(folder / 'หน้า1.jpg', np.full((20, 20), 128, np.uint8))
    X, names = load_images(str(folder))
    assert names == ['หน้า1.jpg'] and X.shape[0] == 1
