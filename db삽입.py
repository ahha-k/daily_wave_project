
import pandas as pd
from pymongo import MongoClient

# MongoDB 연결 설정
def connect_to_mongodb(uri, db_name, collection_name):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    return collection

# CSV 파일을 MongoDB에 저장
def csv_to_mongodb(csv_file_path, collection):
    # CSV 파일 읽기
    data = pd.read_csv(csv_file_path)
    # 데이터프레임을 딕셔너리 리스트로 변환
    data_dict = data.to_dict(orient='records')
    # MongoDB에 데이터 삽입
    collection.insert_many(data_dict)
    print(f"{len(data_dict)}개의 문서가 MongoDB에 삽입되었습니다.")

if __name__ == "__main__":
    # CSV 파일 경로
    csv_file_path = "processed_melon_songs.csv"  # CSV 파일 경로를 입력하세요
    # MongoDB URI
    mongodb_uri = "mongodb://localhost:27017/"  # MongoDB URI 설정
    # 데이터베이스와 컬렉션 이름
    database_name = "local"  # 데이터베이스 이름을 입력하세요
    collection_name = "dailywave"  # 컬렉션 이름을 입력하세요

    # MongoDB 컬렉션 연결
    collection = connect_to_mongodb(mongodb_uri, database_name, collection_name)
    
    # CSV 파일을 MongoDB로 저장
    csv_to_mongodb(csv_file_path, collection)
