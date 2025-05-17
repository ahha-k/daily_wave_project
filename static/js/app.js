document.getElementById('emotionForm').addEventListener('submit', async function (e) {
    e.preventDefault(); // 기본 폼 제출 방지
    const userInput = document.getElementById('textInput').value;

    // Fetch 요청
    fetch('/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: userInput })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json(); // JSON 응답 처리
    })
    .then(data => {
        console.log(data); // 디버깅용 콘솔 출력

        // 결과 표시
        const resultArea = document.getElementById('resultArea');
        resultArea.innerHTML = `
            <h3>감정 분석 결과</h3>
            <p>최상위 감정 1: ${data['최상위감정1']} (${data['최상위감정1비율'] * 100}%)</p>
            <p>최상위 감정 2: ${data['최상위감정2']} (${data['최상위감정2비율'] * 100}%)</p>
            <p>최상위 감정 3: ${data['최상위감정3']} (${data['최상위감정3비율'] * 100}%)</p>
            <h3>추천 노래</h3>
            <ul>
                ${data['노래결과'].map(song => `
                    <li>
                        <strong>${song['노래제목']}</strong> - ${song['아티스트']}
                        <img src="${song['앨범표지']}" alt="앨범 표지" width="100">
                        (유사도: ${song['유사도']})
                    </li>
                `).join('')}
            </ul>
        `;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('resultArea').innerHTML = `<p>오류 발생: ${error.message}</p>`;
    });
});
