pipeline {
    agent any
    options {
        skipStagesAfterUnstable()
    }
    environment {
        GIT_CONFIG_NOSYSTEM = '1'
        PYTHONUNBUFFERED = '1'
        DATAROBOT_API_KEY = credentials("datarobot_api_key")
        GITHUB_PAT = credentials('github_pat')
    }
    stages {
        stage('Deploy') {
            steps {
                // use a python environment with: datarobot==3.0.2
                withPythonEnv('Python-3.9') {
                    sh "pip install --upgrade pip"
                    sh "pip install -r jenkins_requirements.txt"
                    sh "python custom_model_e2e.py"
                }
            }
        }
        stage('Update GIT') {
            steps {
                script {
                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                        sh "git add log/"
                        sh "git add model_documentation/"
                        sh "git commit --allow-empty -m 'Update logs after Jenkins pipeline build #${env.BUILD_NUMBER}'"
                        sh "git push -f https://${GITHUB_PAT}@github.com/zmaalgorithmia/jenkins_custom_model_e2e.git HEAD:main"
                    }
                }
            }
        }
    }
}