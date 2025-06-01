import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

csv_file = "balanced_sample_test.csv"

# 유형별 숫자 ID 수집 구조
type_to_numbers = defaultdict(list)

# CSV 파싱 및 유형-숫자 분류
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 4:
            continue
        try:
            news_id = row[-1].strip()
            prefix, _, num_str = news_id.split("_")
            number = int(num_str)
            type_to_numbers[prefix].append(number)
        except Exception:
            continue

# 각 유형별로 자동 범위 나눠서 분포 시각화
for prefix, numbers in type_to_numbers.items():
    if not numbers:
        continue

    min_val = min(numbers)
    max_val = max(numbers)
    range_span = max_val - min_val

    # bin 크기 및 개수 결정
    ideal_bin_size = max(range_span // 15, 1000)
    bins = np.arange(min_val, max_val + ideal_bin_size, ideal_bin_size)
    hist, bin_edges = np.histogram(numbers, bins=bins)

    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1]-1)}" for i in range(len(bin_edges)-1)]

    # 텍스트 출력
    print(f"\n=== {prefix} 유형 문서 분포 ===")
    for label, count in zip(bin_labels, hist):
        print(f"{label}: {count}")

    # 그래프 출력
    plt.figure(figsize=(10, 5))
    plt.bar(bin_labels, hist, color='teal')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"{prefix} type ID ")
    plt.xlabel("숫자 범위")
    plt.ylabel("문서 수")
    plt.tight_layout()
    plt.show()
