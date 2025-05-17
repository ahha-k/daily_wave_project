import React from 'react';

const DailyWaveFooter = () => {
  return (
    <footer className="dailywave-footer">
      <div className="music-player">
        <p>Music</p>
        <input type="range" />
      </div>
      <div className="footer-icons">
        <button>👤</button>
        <button>⚙️</button>
      </div>
    </footer>
  );
};

export default DailyWaveFooter;
