<!-- set project -->

gcloud config set project XXXXXXX

<!-- confirm project set -->

gcloud config get-value project

<!-- enable cloud run and cloud build -->

gcloud services enable run.googleapis.com cloudbuild.googleapis.com

<!-- run cloud build -->

gcloud builds submit --config cloudbuild.yaml
