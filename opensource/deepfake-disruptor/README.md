# 딥페이크 생성 방해 AI 모델 💻💬

<br/>

**AI 모델 훈련 및 테스트**
IDE download: [pycharm](https://www.jetbrains.com/pycharm/download/#section=windows)

<br/>

---

<br/>

### 📌 Dependencies

- wandb 설정 (api 키 입력)

```console
wandb init
```

- 실험을 수행하는데 필요한 패키지를 정리한 파일입니다.

```console
pip install -r requirements.txt
```

<br/>

---

<br/>

### 0️⃣ 데이터 전처리 단계 수행

train / test 데이터를 나누고 이미지 특징을 추출하여 저장(.npz)

```console
sh run.sh 0 "데이터가 위치한 폴더"
```

<br/>

### 1️⃣ 모델 훈련

- 미리 훈련된 모델이 없는 경우 (반드시 shell파일 수정 후 사용할 것)

```console
sh run.sh 1 "모델을 저장할 위치"
```

- 미리 훈련된 모델이 있거나 재훈련하는 경우

```console
sh run.sh 1 "훈련된 모델이 저장된 위치" "모델을 저장할 위치"
```

<br/>

### 2️⃣ 모델 추론

훈련된 모델과 테스트할 사진(_.jpg / _.png)을 넣어 테스트

```console
python inference.py
```

---

<br/>

### ⏬ Dataset

- [celebA]
- 다운로드 받아, repository에 있는 ./data/\* 에 압축해제

<br/>

---

<br/>

### License

```
Copyright (c) 2024-deependers.
```
