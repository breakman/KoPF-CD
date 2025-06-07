import csv
import random
from collections import defaultdict

INPUT_CSV = "top25_clickbait_sorted.csv"       # 전체 데이터
OUTPUT_CSV = "balanced_sample_train.csv"   # 출력 파일
SAMPLE_TOTAL = 1000

TYPES = ["EC", "ET", "GB", "IS", "LC", "PO", "SO"]
CLASSES = ["0", "1"]
NUM_GROUPS = len(TYPES) * len(CLASSES)
SAMPLE_PER_GROUP = SAMPLE_TOTAL // NUM_GROUPS  # 285개 정도

# (class, type) 별로 그룹화
grouped_data = defaultdict(list)

with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 4:
            continue

        label = row[0].strip()  # 0 or 1
        news_id = row[-1].strip()
        prefix = news_id.split("_")[0]

        if label in CLASSES and prefix in TYPES:
            grouped_data[(label, prefix)].append(row)

# 그룹별로 균등 샘플 추출
balanced_rows = []
for label in CLASSES:
    for prefix in TYPES:
        key = (label, prefix)
        rows = grouped_data[key]
        if len(rows) < SAMPLE_PER_GROUP:
            raise ValueError(f"그룹 {key}의 데이터가 부족함: {len(rows)}개 존재")
        balanced_rows.extend(random.sample(rows, SAMPLE_PER_GROUP))


# 저장
with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerows(balanced_rows)

print(f"{SAMPLE_TOTAL}개 균형 샘플을 {OUTPUT_CSV}에 저장 완료.")
