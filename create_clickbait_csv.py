import os
import json
import csv
from pathlib import Path

def process_json_files(input_dir):
    # 출력 CSV 파일 경로
    output_file = 'valid.csv'
    
    # CSV 파일 생성 및 헤더 작성
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)  # 모든 필드를 따옴표로 감싸기
        
        # 입력 디렉토리의 모든 JSON 파일 처리
        json_files = list(Path(input_dir).glob('**/*.json'))
        total_files = len(json_files)
        
        print(f'총 {total_files}개의 JSON 파일을 처리합니다...')
        
        for i, json_file in enumerate(json_files, 1):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # labeledDataInfo에서 필요한 정보 추출
                    if 'labeledDataInfo' in data:
                        title = data['labeledDataInfo'].get('newTitle', '')
                        clickbait_class = data['labeledDataInfo'].get('clickbaitClass', '')
                        
                        # CSV 파일에 데이터 추가 (클릭베이트 클래스, 제목 순서로)
                        writer.writerow([clickbait_class, title])
                
                # 진행 상황 출력
                if i % 1000 == 0:
                    print(f'진행률: {i}/{total_files} 파일 처리 완료')
                    
            except Exception as e:
                print(f'파일 {json_file} 처리 중 오류 발생: {str(e)}')
                continue
    
    print(f'\n처리가 완료되었습니다. 결과가 {output_file}에 저장되었습니다.')

def main():
    # 사용자로부터 입력 디렉토리 경로 받기
    while True:
        input_dir = input('JSON 파일이 있는 디렉토리 경로를 입력하세요: ').strip()
        
        if os.path.isdir(input_dir):
            break
        else:
            print('유효하지 않은 디렉토리 경로입니다. 다시 시도해주세요.')
    
    # JSON 파일 처리 시작
    process_json_files(input_dir)

if __name__ == '__main__':
    main() 