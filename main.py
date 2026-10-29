from data import get_data 
from operation import Custering

def main():
    folder_path = "Dataset"
    data_x, file_names = get_data(folder_path)
    print("Data shape:", data_x.shape)

    clusting = Custering()
    result_hardmean = clusting.kmeans_calculate(data_x, k=3, mode='manhattan')
    print("Hard Mean Clustering Results:")
    for name, label in zip(file_names, result_hardmean):
        print(f"Name: {name}, Cluster: {label}")
   
    clusting.group_custering(file_names, result_hardmean, folder_path, "Output")


if __name__ == "__main__":
    main()