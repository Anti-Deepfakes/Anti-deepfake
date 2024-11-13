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
                        sh 'docker-compose stop fastapi-detect'
                        sh 'docker-compose rm fastapi-detect'
                        sh 'docker-compose rmi fastapi-detect'

                        sh 'docker-compose up -d fastapi-detect'
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
                    sh 'docker-compose stop fastapi-detect'
                    sh 'docker-compose rm fastapi-detect'
                    sh 'docker-compose rmi fastapi-detect'
                }
            }
        }
    }
}
