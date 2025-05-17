import pandas as pd
import numpy as np
import json
from pymongo import MongoClient

ID2LABEL_KOR = {0: '분노',
        1: '슬픔',
        2: '불안',
        3: '상처',
        4: '당황',
        5: '기쁨'
        }

def cosine_similarity(vec1, vec2):
    # 벡터의 내적 계산
    dot_product = np.dot(vec1, vec2)
    # 벡터 크기 계산
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # 코사인 유사도 계산
    if norm_vec1 == 0 or norm_vec2 == 0:  # 벡터 크기가 0일 경우 0 반환
        return 0
    return dot_product / (norm_vec1 * norm_vec2)

def get_database_connection():
    # MongoDB 클라이언트 생성
    client = MongoClient("mongodb://localhost:27017/")  # URI를 적절히 설정
    db = client["dailywave"]  # 데이터베이스 이름
    return db["melonsongs"]  # 컬렉션 이름 반환

def calculation(emotion_scores, top_n=5):
    # 상위 감정 추출
    sorted_emotions = sorted(
        [(ID2LABEL_KOR[i], score) for i, score in enumerate(emotion_scores)],
        key=lambda x: x[1],
        reverse=True,
    )
    top_emotions = [{"감정": emo, "비율": round(ratio, 2)} for emo, ratio in sorted_emotions[:3]]

    # MongoDB에서 데이터 로드
    collection = get_database_connection()
    cursor = collection.find({}, {"song_no": 1, "title": 1, "artist": 1, "score": 1})
    data = list(cursor)

    # MongoDB 데이터를 DataFrame으로 변환
    df = pd.DataFrame(data)
    
    # score 필드 변환 (JSON 문자열 -> 리스트)
    df["score"] = df["score"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)


    # 유사도 계산 (1차원 배열로 전달)
    df["similarity"] = df["score"].apply(lambda x: cosine_similarity(x, emotion_scores))

    # 유사도 기준으로 상위 n개 선택
    top_df = df.nlargest(top_n, "similarity")

    # 노래 결과 리스트 생성
    song_results = []
    for _, row in top_df.iterrows():
        song_no = row["song_no"]
        album_cover_path = f"/static/images/{song_no}.jpg"
        song_results.append({
            "노래제목": row["title"],
            "아티스트": row["artist"],
            "앨범표지": album_cover_path,
            "유사도": round(row["similarity"], 4),
        })

    # 반환 데이터 생성
    result = {
        "최상위감정1": top_emotions[0]["감정"],
        "최상위감정1비율": top_emotions[0]["비율"],
        "최상위감정2": top_emotions[1]["감정"],
        "최상위감정2비율": top_emotions[1]["비율"],
        "최상위감정3": top_emotions[2]["감정"],
        "최상위감정3비율": top_emotions[2]["비율"],
        "노래결과": song_results,
    }

    return result

if __name__ == "__main__":
    # result 배열
    result = [0.1345, 0.25432, 0.3123234, 0.14556, 0.2223, 0.1211]
    top_n = 1
    df_with_similarity = calculation(result, top_n=top_n)
    print(df_with_similarity)