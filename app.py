from flask import Flask, render_template, request, jsonify, send_from_directory
from model_predict import predict_score  
from Calculation import calculation     
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pymongo import MongoClient
from flask_cors import CORS


# Flask 앱 초기화
app = Flask(__name__, 
            static_folder='static', 
            template_folder='templates')
CORS(app)
# MongoDB 클라이언트 설정
client = MongoClient('mongodb://localhost:27017/')  # 로컬 MongoDB
db = client['dailywave']  # 데이터베이스 이름
collection = db['diary']  # 컬렉션 이름
# 홈 페이지 라우트
@app.route('/')
def index():
    return render_template('index.html')

# 일기 페이지 라우트
@app.route('/diary_page')
def diary_page():
    date = request.args.get('date', '??/??')
    return render_template('diary_page.html', date=date)

# 마이 페이지 라우트 (체크! DB 데이터 가공 후 템플릿에 넘기려고함) -> 추가 : 정렬기능
@app.route('/mypage')
def mypage():
    entries = list(collection.find())

    def parse_date(entry):
        year = entry.get('year', 0)
        date_str = entry.get('date', '0/0')
        try:
            # date가 'YYYY-MM-DD' 형식이면 split('-')로
            if '-' in date_str:
                parts = date_str.split('-')
                y = int(parts[0])
                m = int(parts[1])
                d = int(parts[2])
            else:
                m, d = map(int, date_str.split('/'))
                y = int(year)
        except:
            y, m, d = 0, 0, 0
        return (y, m, d)

    entries.sort(key=parse_date)

    processed_entries = []
    for entry in entries:
        result = entry.get('result', {})
        emotions = {
            'emotion1': result.get('최상위감정1', ''),
            'percent1': round(result.get('최상위감정1비율', 0) * 100),
            'emotion2': result.get('최상위감정2', ''),
            'percent2': round(result.get('최상위감정2비율', 0) * 100),
            'emotion3': result.get('최상위감정3', ''),
            'percent3': round(result.get('최상위감정3비율', 0) * 100),
        }

        song_info = result.get('노래결과', [{}])[0]
        music = {
            'title': song_info.get('노래제목', ''),
            'artist': song_info.get('아티스트', '')
        }

        processed_entries.append({
            'year': entry.get('year', ''),
            'date': entry.get('date', ''),
            'emotions': emotions,
            'music': music
        })

    return render_template('mypage.html', entries=processed_entries)



# 정적 파일 제공
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# 감정 분석 및 처리 라우트
@app.route('/process', methods=['POST'])
def process_text():
    try:
        # JSON 데이터 받기
        data = request.json
        text = data.get('text', '')  # 입력 텍스트 (없으면 기본값 '')

        if not text.strip():
            return jsonify({'error': '입력된 텍스트가 없습니다.'}), 400

        ckpt_path = './10thou(6l)-4579'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

        # 감정 점수 예측 및 결과 계산
        emotion_scores = predict_score(text, model, tokenizer, device)
        result_data = calculation(emotion_scores, 1)

        print("Result Data:", result_data)  # 디버깅 로그 출력

        # 결과를 JSON 형태로 반환
        return jsonify(result_data)

    except Exception as e:
        # 에러 발생 시 디버깅 정보와 함께 반환
        return jsonify({'error': '처리 중 오류가 발생했습니다.', 'details': str(e)}), 500

@app.route('/handle_entry', methods=['POST'])
def handle_entry():
    try:
        # JSON 데이터 받기
        data = request.json
        print(data)
        #추가
        year = data.get('year', '') # 일단 연도 정보 임의로 추가
        date = data.get('date', '')  # 날짜 정보
        text = data.get('text', '')  # 일기 내용
        full_data = data.get('data', {})  # 전체 JSON 데이터

        if not date.strip():
            return jsonify({'error': '날짜가 필요합니다.'}), 400

        # 삭제: 텍스트가 비어 있으면 해당 날짜 데이터 삭제
        if not text.strip():
            result = collection.delete_one({'date': date})
            if result.deleted_count > 0:
                return jsonify({'message': f'{date}의 데이터가 삭제되었습니다.'})
            else:
                return jsonify({'message': f'{date}의 데이터를 찾을 수 없습니다.'}), 404

        # 데이터 검색
        existing_entry = collection.find_one({'date': date})

        if existing_entry:
            # 수정: 데이터가 존재하면 전체 데이터를 업데이트
            collection.update_one(
                {'date': date},
                {'$set': {'text': text, 'result': full_data}}
            )
            return jsonify({'message': f'{date}의 데이터가 업데이트되었습니다.'})
        else:
            # 입력: 데이터가 없으면 새로 추가
            diary_entry = {
                'year' : year,
                'date': date,
                'text': text,
                'result': full_data  # 전체 JSON 데이터 저장
            }
            collection.insert_one(diary_entry)
            return jsonify({'message': f'{date}의 데이터가 새로 추가되었습니다.'})

    except Exception as e:
        return jsonify({'error': '처리 중 오류가 발생했습니다.', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)