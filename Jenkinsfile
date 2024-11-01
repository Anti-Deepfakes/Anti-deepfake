pipeline {
    agent any


    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    dir('react') {

                        docker.build('react-image', '-f Dockerfile .')
                    }
                }
            }
        }


        
        stage('Up Docker Compose') {
            steps {
                script {
                    dir('react') {
                        sh 'docker-compose up -d'
                    }
                }
            }
        }

    }
}

