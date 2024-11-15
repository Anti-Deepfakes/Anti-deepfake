import React, { useEffect, useState } from 'react';
import { useResult } from '../../context/ResultContext';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../../styles/detect/Result.css';

function Result() {
  const location = useLocation();
  const navigate = useNavigate();
  const [selectedOption, setSelectedOption] = useState('');
  const [predictResult, setPredictResult] = useState(false); // true: 딥페이크 | false: 원본
  const { preview, isFake } = location.state || {};
  const [icon, setIcon] = useState('');

  // 탐지 api 요청 상태
  const { state } = useResult();
  const { loading, result, error } = state;

  useEffect(() => {
    if (loading) {
      setIcon('/loading.gif');
    }
    if (result && result.conf) {
      console.log('확률:', result.conf);
      if (result.conf == 50) { // 원본
        setIcon('/check.png'); 
        setPredictResult(false);
      } else { // 딥페이크 영상물
        setIcon('/warning.png');
        setPredictResult(true);
      }
    }
    if (error) {
      setIcon(null);
    }
  }, [result, loading, error]); // result 값이 바뀔 때만 실행

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
        {result && (
          <div className="result-box">
            {/* predictResult 값에 따라 텍스트 표시 */}
            <p>
              해당 영상물은 {predictResult ? "딥페이크 생성물 " : "원본 "} 입니다.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default Result;
