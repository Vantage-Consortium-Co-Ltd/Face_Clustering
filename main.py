"""
main.py – จุดเริ่มต้น: โหลดรูป → K-Means → จัดกลุ่ม → แสดงผล

การเปลี่ยนแปลงจากต้นฉบับ:
- แก้ typo ทั้งหมด: Custering → Clustering, clusting → clustering
- แก้ import: Visualization ถูกลบแล้ว → import visualize function ตรง ๆ
- ลบ comment "Hard Mean Clustering" ที่สับสน (K-Means ไม่เรียกว่า Hard Mean)
- เพิ่ม blank line ระหว่าง import และ function (PEP 8)
"""
from data import get_data
from operation import Clustering
from visualize import visualize


def main() -> None:
    folder_path = "Dataset"
    output_path = "Output"

    data_x, file_names = get_data(folder_path)
    print(f"โหลดรูป {len(file_names)} ไฟล์  shape: {data_x.shape}")

    clustering = Clustering(random_state=42)

    result = clustering.kmeans(data_x, k=3, mode="manhattan")
    print("\nผลการ Clustering:")
    for name, label in zip(file_names, result):
        print(f"  {name}  →  cluster {label}")

    clustering.group_clustering(file_names, result, folder_path, output_path)
    print(f"\nบันทึกรูปไว้ที่ {output_path}/")

    visualize(output_path)


if __name__ == "__main__":
    main()
