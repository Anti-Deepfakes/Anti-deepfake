pipeline {
    agent any

    environment {
        COMPOSE_DIR = '/home/ubuntu/dpg/compose/fastapi-detect'
        REPO_DIR = '/var/jenkins_home/workspace/fastapi-detect'
    }

    stages {
        stage('Checkout from Git') {
            steps {
                script {
                    dir("${REPO_DIR}") {
                        // 최신 Git 리포지토리에서 코드 가져오기
                        sh 'git reset --hard' // 변경 사항 초기화
                        sh 'git pull origin develop' // 최신 코드 가져오기 (develop 브랜치 기준)
                    }
                }
            }
        }

        stage('Build Docker Image for FastAPI') {
            steps {
                script {
                    dir('./app') {
//                             sh 'docker stop fastapi_detect || true'
//                             sh 'docker rm fastapi_detect || true'
//                             sh 'docker rmi fastapi-detect:latest || true'

                        sh 'docker build -t fastapi-detect:latest -f Dockerfile .'
                    }
                }
            }
        }

        stage('Deploy FastAPI with Docker Compose') {
            steps {
                script {
                    dir("${REPO_DIR}") {
                        // 최신의 docker-compose.yml 파일을 사용하여 컨테이너를 배포
                        dir("${COMPOSE_DIR}") {
                            sh 'docker-compose build --no-cache'
                            sh 'docker-compose up -d'
                        }
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
//                     sh 'docker stop fastapi_detect || true'
//                     sh 'docker rm fastapi_detect || true'
//                     sh 'docker rmi fastapi-detect:latest || true'
                }
            }
        }
    }
}
