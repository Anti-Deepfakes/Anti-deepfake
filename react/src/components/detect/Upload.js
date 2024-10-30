import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import '../../styles/detect/Upload.css';


function Upload() {
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false); // 요청 상태 관리
  const navigate = useNavigate();

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e) => e.preventDefault();

  const handleJudge = async () => {
    if (!preview) {
      alert('이미지를 업로드해주세요!');
      return;
    }

    setLoading(true); // 요청 시작
    try {
      // TODO [강윤서] : 탐지 fast api 연결
      const response = await axios.get('https://www.naver.com');
      console.log('네이버 응답:', response.data);
    } catch (error) {
      console.error('네이버 요청 중 오류 발생:', error);
    } finally {
      setLoading(false); // 요청 종료
      navigate('/detect/predict', { state: { preview } });
    }
  };

  return (
    <div className="main-container">
      <div className="navy-banner">
        <h2>딥페이크 영상물인지 판단하고 싶은 영상을 업로드해주세요.</h2>
      </div>

      <div className="upload-container">
        <div
          className="upload-box"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <p>Drag<br />업로드</p>
          <input type="file" onChange={handleImageUpload} />
        </div>

        {preview && (
          <div className="preview-box">
            <img src={preview} alt="미리보기" />
          </div>
        )}
      </div>

      <button className="judge-button" onClick={handleJudge} disabled={loading}>
        {loading ? '요청 중...' : '판단하기'}
      </button>
    </div>
  );
}

export default Upload;
