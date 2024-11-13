pipeline {
    agent any

    environment {
        COMPOSE_DIR = '/home/ubuntu/dpg/compose/fastapi-detect'
    }

    stages {
        stage('Build Docker Image for FastAPI') {
            steps {
                script {
                    // FastAPI Docker 이미지 빌드
                    dir('./app') {
                        sh 'ls -al' // Dockerfile이 있는지 확인
                        docker.build('fastapi-detect', '-f Dockerfile .')
                    }
                }
            }
        }

        stage('Deploy FastAPI with Docker Compose') {
            steps {
                script {
                    dir("${COMPOSE_DIR}") {
                        // 기존 컨테이너 제거 및 이미지 삭제
                        sh 'docker stop fastapi_detect || true'
                        sh 'docker rm fastapi_detect || true'
                        sh 'docker rmi fastapi-detect || true'

                        // Docker Compose로 FastAPI 컨테이너 빌드 및 실행
                        sh 'docker-compose down || true'
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
                    sh 'docker stop fastapi_detect || true'
                    sh 'docker rm fastapi_detect || true'
                    sh 'docker-compose down || true'
                    sh 'docker rmi fastapi-detect || true'
                }
            }
        }
    }
}
