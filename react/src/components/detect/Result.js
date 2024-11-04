// src/components/Result.js
import React, { useEffect, useState } from 'react';
import { useResult } from '../../context/ResultContext';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../../styles/detect/Result.css';

function Result() {
  const location = useLocation();
  const navigate = useNavigate();
  const [selectedOption, setSelectedOption] = useState('');
  const { preview } = location.state || {};
  const [icon, setIcon] = useState('');

  // 탐지 api 요청 상태
  const { state } = useResult();
  const { loading, result, error } = state;

  useEffect(() => {
    if (loading) {
      setIcon('/loading.gif');
    }
    if (result && result.result) {
      console.log('확률:', result.result);
      if (result.result > 50) { // 딥페이크 영상물
        setIcon('/warning.png');
      } else { // 원본
        setIcon('/check.png'); 
      }
    }
  }, [result]); // result 값이 바뀔 때만 실행

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // TODO 피드백 post api 만들기
    } catch (error) {
      console.error('오류 발생:', error);
      alert('오류가 발생했습니다. 나중에 다시 시도해주세요.');
    }
  };

  return (
    <div className="result-container">
      <div className="title">
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
        {/* 상태에 따라 내용 표시 */}
        {/* {loading && <p>처리 중입니다...</p>} */}
        {error && <p className="error-text">{error}</p>}
        {result && result.result && (
          <div className="result-box">
            <p>해당 영상물은 {result.result}% 확률로 딥페이크 영상물입니다.</p>
          </div>
        )}
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
