import os
import csv
from OpenPrompt.openprompt.data_utils.utils import InputExample
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgnewsTitleProcessor:
    def __init__(self):
        self.labels = ["clickbait", "not-clickbait"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.csv")
        examples = []

        with open(path, encoding='utf-8-sig') as f:  # BOM 제거
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for idx, row in enumerate(reader):
                if len(row) != 2:
                    logger.warning(f"Invalid row format at index {idx}: {row}")
                    continue

                label, title = row

                title = title.translate(str.maketrans({
                    '“': '"', '”': '"',
                    '‘': "'", '’': "'"
                }))

                text_a = title.replace('\\', ' ').strip()

                try:
                    label = int(label)
                    if label not in [0, 1]:
                        logger.warning(f"Invalid label at index {idx}: {label}")
                        continue
                except ValueError:
                    logger.warning(f"Invalid label format at index {idx}: {label}")
                    continue

                example = InputExample(
                    guid=str(idx),
                    text_a=text_a,
                    label=label
                )
                examples.append(example)

        return examples



# 테스트 실행
if __name__ == "__main__":
    processor = AgnewsTitleProcessor()
    dataset_path = "./datasets/TextClassification/KR-Clickbait"  # 예시 경로
    examples = processor.get_examples(dataset_path, "train")

    for ex in examples[:12]:
        print(f"GUID: {ex.guid}")
        print(f"Label: {ex.label}")
        print(f"Title: {ex.text_a}")
        print("="*40)
