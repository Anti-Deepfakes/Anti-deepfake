# 1. 빌드 단계
FROM node:20 AS builder

# 작업 디렉토리 설정
WORKDIR /app

# package.json과 package-lock.json을 복사해 의존성 설치
COPY package.json ./
RUN npm install

# 소스 코드 전체 복사 및 빌드 수행
COPY . .
RUN npm run build

# 2. 실행 단계
FROM nginx:stable-alpine

# Nginx 설정 파일을 덮어씌움
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 빌드 결과물인 정적 파일을 Nginx가 서빙하는 디렉토리로 복사
COPY --from=builder /app/build /usr/share/nginx/html

# Nginx 기본 포트
EXPOSE 80

# Nginx 실행
CMD ["nginx", "-g", "daemon off;"]