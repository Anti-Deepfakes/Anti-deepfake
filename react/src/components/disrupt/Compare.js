import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import '../../styles/disrupt/Compare.css';
function Compare() {
    const navigate = useNavigate();
    const location = useLocation();
    const preview = location.state?.preview;
    return (
        <div className="model-container">
            <button onClick={() => navigate('/')} className="home-button">
                홈으로 돌아가기
            </button>
            <div>원본 이미지</div>
          <div className="image-row">
            <img src={preview} alt="Input Face 1" className="face-image" />
            <div className="arrow">→</div>
            <div className="model-block">딥페이크 생성 모델</div>
            <div className="arrow">→</div>
            <img src={preview} alt="Output Face 1" className="face-image" />
          </div>
          <div>노이즈 삽입 이미지</div>
          <div className="image-row">
            <img src={preview} alt="Input Face 2" className="face-image" />
            <div className="arrow">→</div>
            <div className="model-block">딥페이크 생성 모델</div>
            <div className="arrow">→</div>
            <img src={preview} alt="Output Face 2" className="face-image" />
          </div>
    
        </div>
      );
}

export default Compare;
