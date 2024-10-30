// src/components/Home.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/Home.css';

function Home() {
  const [currentPage, setCurrentPage] = useState(1);
  const navigate = useNavigate();

  const handleNextPage = () => {
    setCurrentPage((prevPage) => (prevPage === 1 ? 2 : 1));
  };

  const handlePrevPage = () => {
    setCurrentPage((prevPage) => (prevPage === 1 ? 2 : 1));
  };

  return (
    <div className="container">
      <div className={`content-wrapper slide-${currentPage}`}>
        <div className={`content ${currentPage === 1 ? 'active' : ''}`}>
          <div className="navy-banner">
            <h2>TruthGuard</h2>
            <p className="subtitle">“가짜가 아닌 진짜를 지키다”</p>
            <p>
              TruthGuard는 머신러닝 기술을 활용하여 영상을 분석하고 해당 콘텐츠가 딥페이크인지 여부를 판단하는 서비스입니다.
            </p>
            <div className="buttons">
              <button onClick={() => navigate('/disrupt/upload')}>생성 방해</button>
              <button onClick={() => navigate('/detect/upload')}>탐지</button>
            </div>
          </div>

          {/* 화살표 네비게이션 */}
          <div className="arrow-container">
            <button className="arrow left-arrow" onClick={handlePrevPage}>
              &#9664;
            </button>
            <button className="arrow right-arrow" onClick={handleNextPage}>
              &#9654;
            </button>
          </div>
        </div>

        <div className={`content ${currentPage === 2 ? 'active' : ''}`}>
          <div className="navy-banner">
            <h2>TruthGuard</h2>
            <p>
              딥페이크 기술의 악용으로 허위 정보와 사기 등이 확산되고 있습니다.
              TruthGuard는 신뢰할 수 있는 정보를 제공하며, 프라이버시를 보호하는 것을 목표로 합니다.
            </p>
            <div className="buttons">
              <button>생성 방해</button>
              <button onClick={() => navigate('/main')}>탐지</button>
            </div>
          </div>

          {/* 화살표 네비게이션 */}
          <div className="arrow-container">
            <button className="arrow left-arrow" onClick={handlePrevPage}>
              &#9664;
            </button>
            <button className="arrow right-arrow" onClick={handleNextPage}>
              &#9654;
            </button>
          </div>
        </div>
      </div>
      
      <div className="footer">
        <p>‘안팁페이크’ 프로젝트는 오픈소스 커뮤니티에 공개되어 있습니다.</p>
        <p>누구나 코드에 기여하고 개선에 참여할 수 있습니다.</p>
        <div className="image-container">
          <a href="https://www.naver.com">
            <img src="/github.png" alt="GitHub Logo" className="github-logo" />
          </a>
          <a href="https://www.naver.com">
            <img src="/discord.png" alt="Discord Logo" className="discord-logo" />
          </a>
        </div>
      </div>
    </div>
  );
}

export default Home;
