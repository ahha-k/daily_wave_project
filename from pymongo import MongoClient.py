from pymongo import MongoClient
import pandas as pd
import json
def test_mongo_connection():
    # MongoDB 클라이언트 생성
    client = MongoClient("mongodb://localhost:27017/")  # URI를 적절히 설정
    db = client["local"]  # 데이터베이스 이름
    collection = db["dailywave"]  # 컬렉션 이름

    cursor = collection.find({}, {"song_no": 1, "title": 1, "artist": 1, "score": 1})
    data = list(cursor)

    if not data:
        raise ValueError("MongoDB에서 데이터를 가져오지 못했습니다.")

    # 데이터프레임으로 변환
    df = pd.DataFrame(data)
    
    df["score"] = df["score"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    print(type(df["score"].iloc[0][0]))  # score 리스트의 첫 번째 값 타입 확인 (float여야 함)
test_mongo_connection()