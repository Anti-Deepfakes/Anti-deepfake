pipeline {
    agent any

    stages {

        
        stage('build image') {
            steps {
                script {
                    dir('./'){
                        docker.build('disrupt-train-image', '-f Dockerfile .')
                    }   
                }
            }
        }
        
        stage('docker-compose up') {
            steps {
                script {
                    dir('/var/compose/disrupt/train'){
                        sh 'docker-compose up -d'
                    } 
                }
            }
        }
        

    }
}