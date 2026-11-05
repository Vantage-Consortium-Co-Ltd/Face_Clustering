# Face_Custering

Face image clustering implemented from scratch with NumPy (hard K-means). The images are grouped into one output folder per cluster, and each cluster is shown in its own matplotlib window.

## How it works

1. `get_data` reads every `.jpg` in the dataset folder as grayscale, resizes it to `IMG_SIZE` (default 240x300) and flattens it into one vector per image. It returns the vectors together with the file names, in the same order.
2. `Custering.kmeans_calculate` runs K-means on those vectors (random initial centroids, distance `mode='manhattan'` or `'square_euclidean'`, both averaged over pixels), prints the centroid change on every iteration until it converges, and returns the cluster index of each image.
3. `Custering.group_custering` reads each original image from the dataset folder with OpenCV and writes it to `Output/cluster_<n>/`, where `<n>` is the cluster index of that image. The output folder is deleted and recreated on every run, so it only ever holds the latest result.
4. `Visualization.visualize` opens one matplotlib window per `cluster_<n>` folder, all at once, with every image in that cluster in a grid so the clusters can be compared side by side.

## Project structure

```
Face_Custering/
├── Dataset/                 # face images (git-ignored, create it yourself)
│   ├── example.jpg
│   ├── example_2.jpg
│   └── ...
├── Output/                  # clustered images (git-ignored, created on run)
│   ├── cluster_0/
│   ├── cluster_1/
│   └── ...
├── main.py                  # entry point
├── data.py                  # image loading and preprocessing
├── operation.py             # clustering and output grouping
├── visualize.py             # one matplotlib window per cluster
├── requirements.txt         # Python dependencies
└── README.md
```

| File | Description |
|---|---|
| `main.py` | Entry point: load data, run K-means, print `Name: <file>, Cluster: <n>`, copy the images into `Output/` and show the clusters |
| `data.py` | `get_data` (load and preprocess images, returns `data_x, file_names`) |
| `operation.py` | `Custering` class: `kmeans_calculate` (K-means) and `group_custering` (clear the output folder, then write images into cluster folders) |
| `visualize.py` | `Visualization` class: `visualize` (one window per cluster folder, `cols` images per row) |
| `requirements.txt` | Python dependencies |

## Dataset layout

Put the face images directly inside `Dataset/`, next to `main.py`. Only `.jpg` files are read, and subfolders are not scanned.

```
Dataset/
├── example.jpg
├── example_2.jpg
└── example_3.jpg
```

The file name is what gets printed next to the cluster number, so name each file after the person or image you want to recognize in the output, for example `example.jpg`. The `Dataset/` and `Output/` folders are git-ignored and are not included in the repository.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Example output:

```
Data shape: (18, 72000)
...
Hard Mean Clustering Results:
Name: example.jpg, Cluster: 0
Name: example_2.jpg, Cluster: 2
...
```

The images are then written to `Output/cluster_0/`, `Output/cluster_1/` and so on, and one window per cluster opens showing its images. Close the windows to end the program.

Change the number of clusters with `k` in `main.py`, the distance with `mode` in `main.py`, the output folder with the last argument of `group_custering`, the images per row in the windows with `cols` in `visualize`, and the image size with `IMG_SIZE` in `data.py`.

## Notes

- All images are resized to the same size, otherwise the flattened vectors have different lengths and `np.array` fails.
- K-means starts from random centroids, so cluster numbers (and sometimes the grouping) can differ between runs.
- An empty cluster keeps its previous centroid.
- Images in `Output/` are re-encoded by OpenCV (`imread` then `imwrite`), so they are saved in color but are not byte-identical copies of the originals.
- `Output/` is deleted at the start of every run, so do not point it at a folder you want to keep (it must also differ from the dataset folder, otherwise `group_custering` raises `ValueError`).
- The windows have no scrollbar: a cluster with many images makes a tall window, so raise `cols` to make it wider and shorter.
- Raw pixels mostly capture lighting and background rather than identity, so clusters may not match people.
- `.heif` and other non-`.jpg` files in `Dataset/` are skipped.
