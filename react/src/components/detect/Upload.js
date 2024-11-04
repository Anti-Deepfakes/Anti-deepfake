import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import '../../styles/detect/Upload.css';
import { useResult } from '../../context/ResultContext';

function Upload() {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const navigate = useNavigate();
  const { dispatch } = useResult();

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
        setFile(file);
      };
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
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e) => e.preventDefault();

  const handleJudge = () => {
    if (!file) {
      alert('이미지를 업로드해주세요!');
      return;
    }

    dispatch({ type: 'REQUEST_START' }); // 로딩 상태 설정
    const formData = new FormData();
    formData.append('image', file);
    // axios.post('https://truthguard.site/model/photo/detection', formData, {
    //   headers: {
    //     'Content-Type': 'multipart/form-data',
    //   },
    // })
    //   .then((response) => {
    //     dispatch({ type: 'REQUEST_SUCCESS', payload: response.data });
    //     console.log(response)
    //   })
    //   .catch((error) => {
    //     console.error(error);
    //     dispatch({ type: 'REQUEST_ERROR', payload: '결과를 가져오는 데 실패했습니다.' });
    //   });
    axios.post('https://truthguard.site/api/photos/detection', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
      .then((response) => {
        dispatch({ type: 'REQUEST_SUCCESS', payload: response.data });
      })
      .catch((error) => {
        console.error(error);
        dispatch({ type: 'REQUEST_ERROR', payload: '결과를 가져오는 데 실패했습니다.' });
      });

    navigate('/detect/predict', { state: { preview } }); // 비동기 요청 중에 페이지 이동
  };

  return (
    <div className="main-container">
      <div className="title">
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

      <button className="judge-button" onClick={handleJudge}>
        판단하기
      </button>
    </div>
  );
}

export default Upload;
