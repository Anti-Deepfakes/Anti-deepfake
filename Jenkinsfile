pipeline {
    agent any

    environment {
        COMPOSE_DIR = '/var/compose'
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
                        sh 'docker-compose down'
                        sh 'docker-compose up -d fastapi-app'
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
                    sh 'docker-compose down'
                }
            }
        }
    }
}
