pipeline {
    agent any

    stages {

        
        stage('build image') {
            steps {
                dir('./ffastapi'){
                    docker.build('fast-image', '-f Dockerfile .')
                }   
            }
        }
        
        stage('docker-compose up') {
            steps {
                script {
                    dir('/var/compose/fastapi'){
                        sh 'docker-compose up -d'
                    } 
                }
            }
        }
        

    }
}
