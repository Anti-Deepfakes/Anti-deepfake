import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';


function Main() {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false); // 요청 상태 관리
  const navigate = useNavigate();

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
      const response = await axios.post("https://truthguard.site/disrupt/disrupt/generate", formData, {
        headers: {
          'Content-Type': "multipart/form-data",
        }
      });
      console.log(response);
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
          <div className="preview-box">
            <img src={preview} alt="미리보기" />
          </div>
        )}
        {loading && (
          <img src="/loading.gif" alt="로딩중" className="result-icon" />
        )}
      </div>

      <button className="judge-button" onClick={handleJudge} disabled={loading}>
        {loading ? '요청 중...' : '삽입하기'}
      </button>
    </div>
  );
}

export default Main;
