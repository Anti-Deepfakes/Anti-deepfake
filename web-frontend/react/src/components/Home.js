// src/components/Home.js
import React from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/Home.css';

function Home() {
  const navigate = useNavigate();

  return (
    <div className="container">
      <h2>안팁페이크(Anti-Deepfake)</h2>
      <div className='`content-wrapper'>
        <div className='content'>
          <div className="navy-banner">
            
            <div className="subtitle">“가짜가 아닌 진짜를 지키다” </div>
            <br />
            <p>
            <span className="highlight">안팁페이크(Anti-Deepfake)</span>는 딥페이크 기술의 악용을 방지하고 <br/>진실성을 보호하는 AI 기반 서비스입니다. </p>
            <br/>
            <p><span className="highlight">생성 방해</span> 기능은 딥페이크 생성 모델의 인식을 방해하는 노이즈를 삽입하여 <br/>악의적 활용을 막습니다. </p>
            <br/>
            <p><span className="highlight">탐지</span> 기능은 업로드된 콘텐츠가 딥페이크인지 여부를 판별하여 <br/>신뢰할 수 있는 정보 제공과 개인정보 보호를 지원합니다.</p>
            <div className="buttons">
              <button onClick={() => navigate('/disrupt/upload')}>생성 방해</button>
              <button onClick={() => navigate('/detect/upload')}>탐지</button>
            </div>
          </div>
        </div>
      </div>
      
      <div className="footer">
        <p>‘안팁페이크’ 프로젝트는 오픈소스 커뮤니티에 공개되어 있습니다.</p>
        <p>누구나 코드에 기여하고 개선에 참여할 수 있습니다.</p>
        <div className="image-container">
          <a href="https://lab.ssafy.com/s11-final/S11P31B201">
            <img src="/gitlab.png" alt="Gitlab Logo" className="gitlab-logo" />
          </a>
         
        </div>
      </div>

    </div>
  );
}

export default Home;
