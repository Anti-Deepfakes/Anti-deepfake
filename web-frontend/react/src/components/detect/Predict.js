import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import '../../styles/detect/Predict.css';

function Predict() {
  const location = useLocation();
  const preview = location.state?.preview;
  const [selectedOption, setSelectedOption] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();

    // 선택되지 않았을 경우 경고 메시지 표시
    if (selectedOption === '') {
      alert('옵션을 선택해주세요.');
      return;
    }

    // 선택된 값을 boolean으로 변환
    const isFake = selectedOption === 'yes';

    // 페이지 이동과 데이터 전달
    navigate('/detect/result', { state: { preview, isFake } });
  };

  return (
    <div className="review-container">
      <div className="title">
        <h2>해당 영상은 딥페이크 제작물 같나요?</h2>
      </div>

      <div className="review-content">
        {preview && <img src={preview} alt="미리보기" className="preview-img" />}

        <form onSubmit={handleSubmit} className="review-form">
          <div className="radio-group">
            <label>
              <input
                type="radio"
                value="yes"
                checked={selectedOption === 'yes'}
                onChange={() => setSelectedOption('yes')}
              />
              예
            </label>
            <label>
              <input
                type="radio"
                value="no"
                checked={selectedOption === 'no'}
                onChange={() => setSelectedOption('no')}
              />
              아니요
            </label>
          </div>
          <button type="submit" className="submit-button">제출</button>
        </form>
      </div>
    </div>
  );
}

export default Predict;
