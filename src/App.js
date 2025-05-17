import React from 'react';
import DailyWaveHeader from '../static/js/DailyWaveHeader'; // 헤더 컴포넌트
import Calendar from './components/Calendar'; // 달력 컴포넌트
import DailyWaveFooter from './components/DailyWaveFooter'; // 푸터 컴포넌트
import './components/App.css'; // App.css 경로 확인 후 수정

function App() {
  return (
    <div className="app-container">
      <DailyWaveHeader />
      <Calendar />
      <DailyWaveFooter />
    </div>
  );
}

export default App;
