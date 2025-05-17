from model_predict import predict_score
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 모델 및 토크나이저 초기화
ckpt_path = './10thou(6l)-4579'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

# 데이터 로드
file_path = 'melon_songs.csv'
df = pd.read_csv(file_path)
df.drop_duplicates(inplace=True)

# 데이터 배치 처리 설정
batch_size = 50  # 한 번에 처리할 데이터 개수
num_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)

# 결과 저장 파일 초기화
output_file = 'processed_' + file_path

if num_batches > 0:
    print(f"총 {num_batches}개의 배치로 나뉘어 처리됩니다.")

# 배치별 처리
for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx].copy()  # 배치 데이터 가져오기

    # 모델 예측 수행
    print(f"Batch {batch_idx + 1}/{num_batches} 처리 중...")
    batch_df['score'] = batch_df['lyrics'].apply(lambda x: predict_score(x, model, tokenizer, device))

    # 결과를 파일에 추가 저장
    if batch_idx == 0:  # 첫 번째 배치일 경우 파일 생성
        batch_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    else:  # 이후 배치일 경우 추가 저장
        batch_df.to_csv(output_file, index=False, mode='a', header=False, encoding="utf-8-sig")

    print(f"Batch {batch_idx + 1}/{num_batches} 저장 완료. ({start_idx + 1} ~ {end_idx})")

# 처리 완료 메시지
print(f"모든 배치 처리가 완료되었습니다. 결과는 {output_file}에 저장되었습니다.")
