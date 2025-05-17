// Pause 버튼 클릭 이벤트
fetch('/api/emotion')
.then(response => response.json())
.then(data => {
  // 감정 결과 데이터 설정
  const emoGroup = document.querySelector('.emo_group .group');
  emoGroup.innerHTML = `
    <div class="text-wrapper-1">${data['최상위감정1']} ${data['최상위감정1비율']}%</div>
    <div class="text-wrapper-2">${data['최상위감정2']} ${data['최상위감정2비율']}%</div>
    <div class="text-wrapper-2">${data['최상위감정3']} ${data['최상위감정3비율']}%</div>
  `;

  // 추천 노래 데이터 설정
  const cdImage = document.getElementById('cdImage');
  const songInfo = document.querySelector('.songInfo');

  const topSong = data['노래결과'][0]; // 첫 번째 추천 노래
  cdImage.src = topSong['앨범표지']; // 앨범 이미지 업데이트
  songInfo.textContent = `${topSong['노래제목']} - ${topSong['아티스트']}`; // 노래 제목 업데이트
})
.catch(error => console.error('데이터 가져오기 실패:', error));
// 날짜를 URL 파라미터로 받기
const urlParams = new URLSearchParams(window.location.search);
const date = urlParams.get("date");
const selectedDateElement = document.getElementById("selectedDate");

if (date) {
  // '2024-11-20' 형식의 문자열을 '-'로 분리
  const [year, month, day] = date.split("-");
  // '11/20' 형식으로 출력
  selectedDateElement.textContent = `${month}/${day}`;
}

// 저장 기능
function saveContent() {
  const moodContent = document.getElementById("moodTextarea").value;
  const date = document.getElementById("selectedDate").textContent.trim();

  // 날짜가 선택되어 있으면 그 날짜에 내용 저장
  if (date) {
    localStorage.setItem(date, moodContent);
  }

}

// 페이지 로드 시 저장된 내용 불러오기
window.onload = function () {
  const date = document.getElementById("selectedDate").textContent.trim();
  if (date) {
    const savedMood = localStorage.getItem(date);
    if (savedMood) {
      document.getElementById("moodTextarea").value = savedMood;
    }
  }

  // 화면 크기 조정 시 SAVE 버튼 위치 조정
  adjustSaveButtonPosition();
};

// 화면 크기 조정 시 SAVE 버튼 위치 조정
function adjustSaveButtonPosition() {
  const saveButton = document.getElementById("saveButton");
  const screenWidth = window.innerWidth;
  const screenHeight = window.innerHeight;

  // 화면 크기에 따라 버튼 위치 조정
  const buttonWidth = saveButton.offsetWidth;
  const buttonHeight = saveButton.offsetHeight;

  const adjustedLeft = Math.min(20, screenWidth - buttonWidth - 20);
  const adjustedBottom = Math.min(20, screenHeight - buttonHeight - 20);

  saveButton.style.left = `${adjustedLeft}px`;
  saveButton.style.bottom = `${adjustedBottom}px`;
}

// x 버튼 클릭 시 뒤로 가기
function goBack() {
  const moodContent = document.getElementById("moodTextarea").value;
  const date = document.getElementById("selectedDate").textContent.trim();

  if (date) {
    localStorage.setItem(date, moodContent);
  }
  window.history.back();
}
document.querySelector('.pause-button').addEventListener('click', function () {
  const wrapper = document.querySelector('.overlap-wrapper');
  const result = document.querySelector('.result');
  const moodTextarea = document.getElementById('moodTextarea');
  const songInfoWrapper = document.querySelector('.songInfo-wrapper'); 
  const songInfo = document.querySelector('.songInfo');

  // 'expanded'와 'show' 클래스 토글
  wrapper.classList.toggle('expanded');
  result.classList.toggle('show');


  // 사용자가 입력한 텍스트 가져오기
  const text = moodTextarea.value.trim();

  if (!text) {
    alert('텍스트를 입력해주세요.');
    return;
  }

  saveContent(); // 저장 기능 호출

  // 서버로 데이터 전송
  fetch('/process', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text }), // 입력 텍스트를 JSON 형식으로 전송
  })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      if (data.error) {
        alert(`오류: ${data.error}`);
        return;
      }

      // 서버 응답 데이터를 HTML에 반영
      updatePageWithResult(data);

      // 날짜 가져오기
      const date = document.getElementById("selectedDate").textContent.trim();

      // DB에 데이터 저장
      saveToDatabase(date, text, data);

      // 텍스트가 성공적으로 처리되면 songInfo-wrapper를 보이게 함
      songInfoWrapper.style.display = 'flex';  // songInfo-wrapper 보이기
    })
    .catch(error => {
      console.error('Error:', error);
      alert('서버와 통신 중 문제가 발생했습니다.');
    });
});

  
  // 서버 응답 데이터를 HTML에 반영하는 함수
  function updatePageWithResult(data) {
    // 감정 데이터 업데이트
    const emoGroup = document.querySelector('.emo_group .group');
    emoGroup.innerHTML = `
      <div class="text-wrapper-1">${data['최상위감정1']} ${Math.round(data['최상위감정1비율'] * 100)}%</div>
      <div class="text-wrapper-2">${data['최상위감정2']} ${Math.round(data['최상위감정2비율'] * 100)}%</div>
      <div class="text-wrapper-2">${data['최상위감정3']} ${Math.round(data['최상위감정3비율'] * 100)}%</div>
    `;
  
    // 추천 노래 데이터 업데이트
    const cdImage = document.getElementById('cdImage');
    const songInfo = document.querySelector('.songInfo');
  
    const topSong = data['노래결과'][0]; // 첫 번째 추천 노래
    cdImage.src = topSong['앨범표지']; // 앨범 표지 이미지 설정
    songInfo.textContent = `${topSong['노래제목']} - ${topSong['아티스트']}`; // 노래 제목과 아티스트 설정
  }

  function saveToDatabase(date, text, fullData) {
    console.log("Sending data to /handle_entry:", { date, text, data: fullData });
  
    fetch('/handle_entry', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        date,
        text,
        data: fullData, // 감정 분석 결과 포함
      }),
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        if (data.error) {
          alert(`DB 저장 오류: ${data.error}`);
        } else {
          console.log("DB 저장 성공:", data.message);
        }
      })
      .catch(error => {
        console.error('Error while saving to DB:', error);
        alert('데이터베이스 저장 중 오류 발생');
      });
  }
  
  