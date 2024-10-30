// src/components/Result.js
import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../../styles/detect/Result.css';

function Result() {
  const location = useLocation();
  const navigate = useNavigate();
  const [selectedOption, setSelectedOption] = useState('');
  const { result, preview } = location.state || {};
  const [icon, setIcon] = useState('');
  const [randomValue, setRandomValue] = useState(0); // 랜덤 값 상태 관리

  // 랜덤 값 생성 후 아이콘과 랜덤 값 설정
  useEffect(() => {
    const value = Math.floor(Math.random() * 100) + 1; // 1~100 랜덤 값 생성
    setRandomValue(value); // 상태에 랜덤 값 설정
    console.log('랜덤 값:', value);
    if (value > 50) {
      setIcon('/warning.png'); // 51 이상이면 warning.png
    } else {
      setIcon('/check.png'); // 50 이하이면 check.png
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.get('https://jsonplaceholder.typicode.com/todos/1');
      console.log('응답 데이터:', response.data);
      navigate('/result', { state: { result: response.data, preview } }); // 응답과 미리보기 전달
    } catch (error) {
      console.error('오류 발생:', error);
      alert('오류가 발생했습니다. 나중에 다시 시도해주세요.');
    }
  };

  return (
    <div className="result-container">
      <div className="navy-banner">
        <h2>판단 결과 확인</h2>
      </div>

      <button onClick={() => navigate('/')} className="home-button">
        홈으로 돌아가기
      </button>

      <div className="result-content">
        <div className="preview-wrapper">
          {preview && (
            <img
              src={preview}
              alt="미리보기"
              className="preview-img"
            />
          )}
          {icon && (
            <img src={icon} alt="결과 아이콘" className="result-icon" />
          )}
        </div>

        <div className="random-value">
          <p>해당 영상물은 {randomValue}% 확률로 딥페이크 영상물입니다.</p>
        </div>
      </div>

      <div className="review-content">
        <form onSubmit={handleSubmit} className="review-form">
          <div className="radio-group">
            <label>
              <input
                type="radio"
                value="yes"
                checked={selectedOption === 'yes'}
                onChange={() => setSelectedOption('yes')}
              />
              맞다
            </label>
            <label>
              <input
                type="radio"
                value="no"
                checked={selectedOption === 'no'}
                onChange={() => setSelectedOption('no')}
              />
              아니다
            </label>
          </div>
          <button type="submit" className="submit-button">제출</button>
        </form>
      </div>
    </div>
  );
}

export default Result;
