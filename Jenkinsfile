pipeline {
    agent any

    environment {
        COMPOSE_DIR = '/home/ubuntu/dpg/compose/fastapi-detect'
        REPO_DIR = '/var/jenkins_home/workspace/fastapi-detect'
    }

    stages {
        stage('Build Docker Image for FastAPI') {
            steps {
                script {
                    dir('./app') {
//                             sh 'docker stop fastapi_detect || true'
//                             sh 'docker rm fastapi_detect || true'
//                             sh 'docker rmi fastapi-detect:latest || true'


                        sh 'docker build -t fastapi-detect:latest -f Dockerfile .'

                        sh 'pwd'

                        sh 'sudo cp ../docker-compose.yml ${COMPOSE_DIR}'

                    }
                }
            }
        }

        stage('Deploy FastAPI with Docker Compose') {
            steps {
                script {
                    dir("${COMPOSE_DIR}") {
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
//                     sh 'docker stop fastapi_detect || true'
//                     sh 'docker rm fastapi_detect || true'
//                     sh 'docker rmi fastapi-detect:latest || true'
                }
            }
        }
    }
}
