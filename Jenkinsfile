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
                        docker.build('fastapi-image', '-f Dockerfile .')
                    }
                }
            }
        }

        stage('Deploy FastAPI with Docker Compose') {
            steps {
                script {
                    dir("${COMPOSE_DIR}") {
                        // 컨테이너 이름을 'fastapi_detect'로 변경
                        sh 'docker stop fastapi_detect || true'
                        sh 'docker rm fastapi_detect || true'
                        sh 'docker rmi fastapi-image || true'

                        // Docker Compose로 FastAPI 컨테이너 실행
                        sh 'docker-compose build'
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
                    sh 'docker rmi fastapi-image || true'
                }
            }
        }
    }
}
