import React, { useEffect, useState } from 'react';
import { useResult } from '../../context/ResultContext';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../../styles/detect/Result.css';

function Result() {
  const location = useLocation();
  const navigate = useNavigate();
  const [selectedOption, setSelectedOption] = useState('');
  const { preview, isFake } = location.state || {};
  const [icon, setIcon] = useState('');

  // 탐지 api 요청 상태
  const { state } = useResult();
  const { loading, result, error } = state;

  useEffect(() => {
    if (loading) {
      setIcon('/loading.gif');
    }
    if (result && result.deepfakeProbability) {
      console.log('확률:', result.deepfakeProbability);
      if (result.deepfakeProbability > 50) { // 딥페이크 영상물
        setIcon('/warning.png');
      } else { // 원본
        setIcon('/check.png'); 
      }
    }
    if (error) {
      setIcon(null);
    }
  }, [result, loading, error]); // result 값이 바뀔 때만 실행

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // 선택되지 않았을 경우 경고 메시지 표시
      if (selectedOption === '') {
        alert('옵션을 선택해주세요.');
        return;
      }

      // JSON 객체 생성
      const requestBody = {
        feedbackId: result.feedbackId,
        userPredict: isFake ? 1 : 0,
        userLabel: selectedOption === "yes" ? 1 : 0
      };

      // PUT 요청 보내기
      await axios.put("https://truthguard.site/api/photos/feedback", requestBody, {
        headers: {
          'Content-Type': 'application/json',
        }
      });

      alert('피드백이 성공적으로 제출되었습니다.');
      // navigate('/'); // 제출 후 홈으로 이동

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
        {error && <p className="error-text">{error}</p>}
        {result && result.deepfakeProbability && (
          <div className="result-box">
            <p>해당 영상물은 {result.deepfakeProbability}% 확률로 딥페이크 영상물입니다.</p>
          </div>
        )}
      </div>
      {result && (
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
              신뢰합니다
            </label>
            <label>
              <input
                type="radio"
                value="no"
                checked={selectedOption === 'no'}
                onChange={() => setSelectedOption('no')}
              />
              신뢰하지 않습니다
            </label>
          </div>
          <button type="submit" className="submit-button">제출</button>
        </form>
      </div>
    )}
    </div>
  );
}

export default Result;
