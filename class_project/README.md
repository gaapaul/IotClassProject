Google Cloud Project for ENGR 453

cd iot/api-client/end_to_end_example
virtualenv env && source env/bin/activate
pip install -r requirements.txt

gcloud config set project <your-project-id>

***Make a pubsub topic***
gcloud pubsub topics create tour-pub --project="${DEVSHELL_PROJECT_ID:-Cloud Shell}"
gcloud pubsub subscriptions create tour-sub --topic=tour-pub

***Make a registry***
gcloud iot registries create tour-registry \
  --region=us-central1 --event-notification-config=topic=tour-pub

***Make a Key***
openssl req -x509 -newkey rsa:2048 -days 3650 -keyout rsa_private.pem \
    -nodes -out rsa_public.pem -subj "/CN=unused"

***Make a device***
gcloud iot devices create test-dev --region=us-central1 \
  --registry=tour-registry \
  --public-key path=rsa_public.pem,type=rs256

***Make a second device***
gcloud iot devices create test-dev2 --region=us-central1 \
  --registry=tour-registry \
  --public-key path=rsa_public.pem,type=rs256

***Add cloud function***
run scripts on devices 
use scripts
