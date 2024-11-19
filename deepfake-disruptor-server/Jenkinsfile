
pipeline {
    agent any

    stages {

        
        stage('build image') {
            steps {
                script {
                    dir('./app'){
                        docker.build('disrupt-server-image', '-f Dockerfile .')
                    }   
                }
            }
        }
        
        stage('docker-compose up') {
            steps {
                script {
                    dir('/var/compose/disrupt_server'){
                        sh 'docker-compose up -d'
                    } 
                }
            }
        }
        

    }
}
