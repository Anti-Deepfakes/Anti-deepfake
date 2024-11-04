// import React from 'react';
// import { useLocation, useNavigate } from 'react-router-dom';
// import '../../styles/disrupt/Compare.css';
// import axios from 'axios';
// function Compare() {
//     const navigate = useNavigate();
//     const location = useLocation();
//     const { preview, disruptedImage } = location.state || {};
//     useEffect(async () => {
//       if (preview && disruptedImage) {
//         const originalFormData = new FormData();
//         originalFormData.append('image', preview);
//         const originalResponse = await axios.post("http://133.186.146.52:8001/model/deepfake/generate", originalFormData, {
//           headers: {
//             'Content-Type': "multipart/form-data",
//           }
//         });
//         const fakeFormData = new FormData();
//         fakeFormData.append('image', disruptedImage);
//         const deepfakeResponse = await axios.post("http://133.186.146.52:8001/model/deepfake/generate", originalFormData, {
//           headers: {
//             'Content-Type': "multipart/form-data",
//           }
//         });
//         console.log(originalResponse);
//         console.log(deepfakeResponse);
//         setIcon('/loading.gif');
//       }
//       if (result && result.result) {
//         console.log(result);
//         // console.log('확률:', result.result);
//         // if (result.result > 50) { // 딥페이크 영상물
//         //   setIcon('/warning.png');
//         // } else { // 원본
//         //   setIcon('/check.png'); 
//         // }
//       }
//       if (error) {
//         setIcon(null);
//       }
//     }, [result, loading, error]); // result 값이 바뀔 때만 실행
//     return (
//         <div className="model-container">
//             <button onClick={() => navigate('/')} className="home-button">
//                 홈으로 돌아가기
//             </button>
//             <div>[ 원본 이미지 ]</div>
//           <div className="image-row">
//             <img src={preview} alt="Input Face 1" className="face-image" />
//             <div className="arrow">→</div>
//             <div className="model-block">딥페이크 생성 모델</div>
//             <div className="arrow">→</div>
//             <img src={preview} alt="Output Face 1" className="face-image" />
//           </div>
//           <div>[ 노이즈 삽입 이미지 ]</div>
//           <div className="image-row">
//             {/* <img src={preview} alt="Input Face 2" className="face-image" /> */}
//             {disruptedImage && (
//             <img src={`data:image/jpeg;base64,${disruptedImage}`} alt="Disrupted Image" className="face-image" />
//           )}
//             <div className="arrow">→</div>
//             <div className="model-block">딥페이크 생성 모델</div>
//             <div className="arrow">→</div>
//             <img src={preview} alt="Output Face 2" className="face-image" />
//           </div>
    
//         </div>
//       );
// }

// export default Compare;


import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../../styles/disrupt/Compare.css';

function Compare() {
  const navigate = useNavigate();
  const location = useLocation();
  const { preview, disruptedImage } = location.state || {};
  
  // 각각의 이미지 로딩 상태 관리
  const [originalLoading, setOriginalLoading] = useState(true);
  const [disruptedLoading, setDisruptedLoading] = useState(true);
  
  const [originalResult, setOriginalResult] = useState(null);
  const [disruptedResult, setDisruptedResult] = useState(null);

  useEffect(() => {
    const fetchOriginalData = async () => {
      try {
        setOriginalLoading(true);
        
        // base64 문자열을 File로 변환
        const originalFile = base64ToFile(preview, 'original.jpg');
        const originalFormData = new FormData();
        originalFormData.append('image', originalFile);
        const originalResponse = await axios.post("http://133.186.146.52:8001/model/deepfake/generate", originalFormData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          }
        });
        setOriginalResult(originalResponse.data.data);
  
      } catch (error) {
        console.error("Error fetching original data:", error);
      } finally {
        setOriginalLoading(false);
      }
    };
  
    const fetchDisruptedData = async () => {
      try {
        setDisruptedLoading(true);
  
        // base64 문자열을 File로 변환
        const disruptedFile = base64ToFile(`data:image/jpeg;base64,${disruptedImage}`, 'disrupted.jpg');
        const disruptedFormData = new FormData();
        disruptedFormData.append('image', disruptedFile);
        const disruptedResponse = await axios.post("http://133.186.146.52:8001/model/deepfake/generate", disruptedFormData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          }
        });
        setDisruptedResult(disruptedResponse.data.data);
  
      } catch (error) {
        console.error("Error fetching disrupted data:", error);
      } finally {
        setDisruptedLoading(false);
      }
    };
  
    if (preview) fetchOriginalData();
    if (disruptedImage) fetchDisruptedData();
  
  }, [preview, disruptedImage]);
  
  function base64ToFile(base64Data, filename) {
    const byteString = atob(base64Data.split(',')[1]);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    const blob = new Blob([ab], { type: 'image/jpeg' });
    return new File([blob], filename, { type: 'image/jpeg' });
  }
    

  return (
    <div className="model-container">
      <button onClick={() => navigate('/')} className="home-button">
        홈으로 돌아가기
      </button>
      
      <div>[ 원본 이미지 ]</div>
      <div className="image-row">
        <img src={preview} alt="Input Face 1" className="face-image" />
        <div className="arrow">→</div>
        <div className="model-block">딥페이크 생성 모델</div>
        <div className="arrow">→</div>
        {originalLoading ? (
          <img src="/loading.gif" alt="Loading" className="result-icon" />
        ) : (
          originalResult && <img src={`data:image/jpeg;base64,${originalResult}`} alt="Output Face 1" className="face-image" />
        )}
      </div>

      <div>[ 노이즈 삽입 이미지 ]</div>
      <div className="image-row">
        <img src={`data:image/jpeg;base64,${disruptedImage}`} alt="Disrupted Image" className="face-image" />
        <div className="arrow">→</div>
        <div className="model-block">딥페이크 생성 모델</div>
        <div className="arrow">→</div>
        {disruptedLoading ? (
          <img src="/loading.gif" alt="Loading" className="result-icon" />
        ) : (
          disruptedResult && <img src={`data:image/jpeg;base64,${disruptedResult}`} alt="Output Face 2" className="face-image" />
        )}
      </div>
    </div>
  );
}

export default Compare;
