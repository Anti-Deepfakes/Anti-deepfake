pipeline {
    agent any

    environment {
        COMPOSE_DIR = '/var/compose'
    }

    stages {
        stage('Build Spring Backend') {
            steps {
                script {
                    dir('./test') {
                        sh 'chmod +x gradlew'
                        sh './gradlew build -x test'
                    }
                }
            }
        }

        stage('Build Docker Image for Spring') {
            steps {
                script {
                    dir('./test') {
                        docker.build('spring-image', '-f Dockerfile .')
                    }
                }
            }
        }

        stage('Build Docker Image for FastAPI') {
            steps {
                script {
                    dir('./fastapi') {
                        docker.build('fastapi-image', '-f Dockerfile .')
                    }
                }
            }
        }

        stage('docker-compose up') {
            steps {
                script {
                    dir("${COMPOSE_DIR}") {
                        sh 'docker-compose down'
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
                    sh 'docker-compose down'
                }
            }
        }
    }
}