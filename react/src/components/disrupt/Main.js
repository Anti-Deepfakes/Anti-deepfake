import React, { useState, useRef } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import "../../styles/disrupt/Main.css";

function Main() {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false); // 요청 상태 관리
  const navigate = useNavigate();
  const previewRef = useRef(null); // 미리보기 이미지의 참조 생성

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
        setFile(file);
      }
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
        setFile(file);
      }
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
      const formData = new FormData();
      formData.append('image', file);
      const response = await axios.post("https://anti-deepfake.kr/disrupt/disrupt/generate", formData, {
        headers: {
          'Content-Type': "multipart/form-data",
        }
      });
      // console.log(response);
      navigate('/disrupt/compare', { state: { preview, disruptedImage: response.data.data } });
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false); // 요청 종료
    }
  };

  return (
    <div className="main-container">
      <div className="title">
        <h2>사진을 업로드해주세요.</h2>
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
          <div className="preview-box" style={{ position: 'relative' }}>
            {loading && (
              <img
                src="/loading.gif"
                alt="로딩중"
                className="loading-icon"
                style={{
                  height: previewRef.current ? previewRef.current.height : '50px',
                }}
              />
            )}
            <img
              src={preview}
              alt="미리보기"
              ref={previewRef} // 미리보기 이미지에 ref 연결
              className="preview-img"
            />
          </div>
        )}
      </div>

      <button className="judge-button" onClick={handleJudge} disabled={loading}>
        {loading ? '요청 중...' : '삽입하기'}
      </button>
      <div className='footer'>
      <a href="https://chromewebstore.google.com/detail/truthguard/mofjgcmgpdjpiignbhecbgofgnhlbdel?hl=ko">
          <img src="/chrome.png" alt="Chrome Extension" className="chrome-extension-logo" />
        </a>
        <p>사용자는 크롬 익스텐션을 통해 간편하게 사진을 업로드하고, 보호된 이미지를 다운로드할 수 있습니다. </p>
        <p>처리된 이미지는 서버에 저장되지 않으며, 개인정보는 안전하게 보호됩니다. </p>
        <p>SNS 프로필 사진이나 공개 이미지에 노이즈를 삽입해 악의적인 사용을 예방하세요.</p>
      </div>
    </div>
  );
}

export default Main;
