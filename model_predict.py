import time
import torch
import re
import emoji
import pandas as pd
from soynlp.normalizer import repeat_normalize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 감정 ID -> 라벨 매핑
ID2LABEL_KOR = {0: '분노', 1: '슬픔', 2: '불안', 3: '상처', 4: '당황', 5: '기쁨'}

# 전처리 함수
def clean(text):
    pattern = re.compile(r'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    text = pattern.sub(' ', text)
    text = emoji.replace_emoji(text, replace='')
    text = url_pattern.sub('', text)
    return repeat_normalize(text.strip(), num_repeats=2)

# 텍스트 병합 및 분할 함수
def split_and_adjust_text(text, min_length=30, max_length=70):
    segments = [seg.strip() for seg in text.split('\n\n') if seg.strip()]
    adjusted_segments = []
    temp_segment = ""

    for segment in segments:
        if len(segment) < min_length:
            temp_segment += " " + segment
        else:
            if temp_segment:
                adjusted_segments.append((temp_segment + " " + segment).strip())
                temp_segment = ""
            else:
                adjusted_segments.append(segment)
    if temp_segment:
        adjusted_segments.append(temp_segment.strip())

    final_segments = []
    for segment in adjusted_segments:
        if len(segment) > max_length:
            words = segment.split()
            temp = []
            for word in words:
                temp.append(word)
                if len(" ".join(temp)) > max_length:
                    final_segments.append(" ".join(temp))
                    temp = []
            if temp:
                final_segments.append(" ".join(temp))
        else:
            final_segments.append(segment)
    return final_segments

# 추론 함수
def infer(sentences, model, tokenizer, device):
    results = []
    model.eval()
    sigmoid = torch.nn.Sigmoid()

    for sentence in sentences:
        sentence = clean(sentence)
        encoding = tokenizer(sentence, return_tensors='pt').to(device)
        outputs = model(**encoding)
        preds = sigmoid(outputs.logits.squeeze())
        results.append({label: preds[idx].item() for idx, label in ID2LABEL_KOR.items()})
    return pd.DataFrame(results)

# 점수 계산 함수
def predict_score(sentences, model, tokenizer, device):
    segments = split_and_adjust_text(sentences.strip())
    results_df = infer(segments, model, tokenizer, device)
    probabilities = results_df.sum() / results_df.sum().sum()
    print(probabilities)
    return list(probabilities)

if __name__ == '__main__':
    # 모델 및 토크나이저 로드
    ckpt_path = './10thou(6l)-4579'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    # 데이터 예측
    sentences = "오늘은 아무 이유 없이 우울하고 무기력했다. 하루 종일 침대에 누워만 있었다"
    score = predict_score(sentences, model, tokenizer, device)
    print(score)
