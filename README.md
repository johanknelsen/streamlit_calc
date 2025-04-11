# Build and deploy

## Command to build the application. PLease remeber to change the project name and application name

## Project name is the project you decide to set it up as

```
gcloud builds submit --tag gcr.io/<ProjectName>/compound_interest_app.py  --project=compound-interest
```
## Command to deploy the application
```
gcloud run deploy --image gcr.io/<ProjectName>/compound_interest_app.py --platform managed  --project=project=compound-interest --allow-unauthenticated
```
