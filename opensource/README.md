# 안팁페이크
## 기능 설명

본 프로젝트는 딥페이크 범죄 예방을 위한 오픈 소스 도구로, 적대적 예제 생성을 통해 딥페이크 생성 모델을 방해하고, 딥페이크 영상물을 효과적으로 탐지할 수 있는 기능을 제공합니다.

### 주요 기능

#### 1️⃣ 적대적 예제 생성
- **U-Net 기반 모델**을 사용하여 얼굴 이미지에 대한 미세한 노이즈를 추가함으로써, 딥페이크 생성 모델 (예: DeepFaceLab, Face2Face 등)이 변환을 수행하는 데 방해 요소로 작용합니다.
- 이 노이즈는 육안으로 식별이 어려우면서도 딥페이크 모델의 변조 기능을 저해합니다.

  ##### ✔ 노이즈의 특징
  - 이미지의 시각적 품질을 유지하면서 탐지 및 변조 저항성을 극대화합니다.
  - 영상에 적용 시, 자연스러운 결과를 보장하면서도 딥페이크 생성 모델의 성능을 저하시킵니다.

  ##### ✔ 기능 특장점
  - 이미지에 bbox와 landmarks에 가중치를 두어 모델 입력값으로 사용합니다.

#### 2️⃣ 딥페이크 탐지
- **Inception ResNet v2 기반의 탐지 모델**을 통해 딥페이크 여부를 판별합니다.
- 탐지 모델은 다양한 딥페이크 생성 방식을 학습하여 높은 탐지 정확도를 보장합니다.

  ##### ✔ 기능 특장점
  - **68 points Face Landmark** 사용하여 눈, 코, 입을 분리하여 픽셀의 유동을 감지합니다.

### 아키텍처
- **적대적 예제 생성 모델**: U-Net 기반의 노이즈 생성기.
- **딥페이크 탐지 모델**: Inception ResNet v2 네트워크.
- **딥페이크 생성 참고 모델**: DeepFaceLab 및 Face2Face 모델.

### 기대 효과
- 딥페이크 범죄 방지 및 탐지 기능 강화.
- 오픈 소스 도구를 통해 연구 및 실무에서 손쉽게 활용 가능.
