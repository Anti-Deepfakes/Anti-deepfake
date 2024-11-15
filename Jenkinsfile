pipeline {
    agent any

//     environment {
//         COMPOSE_DIR = '/home/ubuntu/dpg/compose/fastapi-detect'
//     }

    stages {
        stage('Build Docker Image for FastAPI') {
            steps {
                script {
                    dir('./app') {
//                             sh 'docker stop fastapi_detect || true'
//                             sh 'docker rm fastapi_detect || true'
//                             sh 'docker rmi fastapi-detect:latest || true'
                        sh 'pwd'
                        sh 'ls -al'

                        sh 'docker build -t fastapi-detect:latest -f Dockerfile .'
                        sh 'docker-compose build --no-cache'
                        sh 'docker-compose up -d'
                    }
                }
            }
        }

//         stage('Deploy FastAPI with Docker Compose') {
//             steps {
//                 script {
//                     dir('$./app') {
//                         sh 'pwd'
//                         sh 'ls'
//                         sh 'docker-compose build --no-cache'
//                         sh 'docker-compose up -d'
//                     }
//                 }
//             }
//         }
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
