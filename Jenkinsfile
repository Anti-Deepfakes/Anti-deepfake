pipeline {
    agent any

    environment {
        // 환경 변수 설정
        COMPOSE_DIR = '/home/ubuntu/dpg/compose/fastapi-detect'
    }

    stages {
        stage('Build Docker Image for FastAPI') {
            steps {
                script {
                    dir('./app') {
                        // 기존 컨테이너 및 이미지 정리 (선택 사항)
//                         sh 'docker stop fastapi-detect || true'
//                         sh 'docker rm fastapi-detect || true'
//                         sh 'docker rmi fastapi-detect:latest || true'

                        // Docker 이미지 빌드
                        sh 'docker build -t fastapi-detect:latest -f Dockerfile .'
                    }
                }
            }
        }

        stage('Deploy FastAPI with Docker Compose') {
            steps {
                script {
                    dir("${COMPOSE_DIR}") {
                        // 최신 docker-compose 파일 복사 (필요 시)
                        sh "cp ../docker-compose.yml ${COMPOSE_DIR}"

                        // Docker Compose 빌드 및 배포
//                         sh 'docker-compose down || true'
                        sh 'docker-compose build --no-cache'
                        sh 'docker-compose up -d'
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'Cleaning up...'
            script {
                dir("${COMPOSE_DIR}") {
                    sh 'docker system prune -f'
                }
            }
        }
    }
}
