// src/components/Predict.js
import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../../styles/detect/Predict.css';

function Predict() {
  const location = useLocation();
  const preview = location.state?.preview;
  const [selectedOption, setSelectedOption] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // const response = await axios.get('https://jsonplaceholder.typicode.com/todos/1');
      // console.log('응답 데이터:', response.data);
      navigate('/detect/result', { state: { preview } }); // 미리보기 이미지 전달
    } catch (error) {
      console.error('오류 발생:', error);
      alert('오류가 발생했습니다. 나중에 다시 시도해주세요.');
    }
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
