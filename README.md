<h2 style="text-align:center;"> MLOps Zoomcamp Project: Email Spam Detection</h2>

<img title="Email Spam Detection Workflow Overview" src="src\images\mlops_overview.png">

<h3>Project Description</h3>

<p>
This project showcases comprehensive data and modeling pipelines, incorporating key MLOps principles. These include data collection, modeling experimentation and tracking, model registry, workflow orchestration, model deployment, and monitoring.
 
</p>

<h3>Problem Statement and Objective</h3>

<p>
Many spam emails still managed to get into inboxes daily. The chore of having to manually report each one of them as spam is repetitive and time wasting. The goal is to create a spam detection system to automate this process by treating the problem as a classification.
</p>

<h3>Steps Overview</h3>
<p>The order of the pipeline is as follows:</p>
<ol>
  <li>Data Collection</li>
  <li>Model Experimentation and Tracking, and Orchestration</li>
  <li>Model Deployment</li>
  <li>Monitoring and Orchestration</li>
</ol>

<h2>Pipeline Steps Execution</h2>
<h3>Data Collection</h3>
<!-- <img title="Data Collection" src="src\image\data collect.png"> -->
<p>
Instead of building the API to fetch emails from my email client, I decided to simulate it using <a href="https://huggingface.co/datasets/Deysi/spam-detection-dataset" target="_blank">Deysi/spam-detection-dataset</a>. The dataset consists of 8,180 train samples and 2,730 test samples. A small subset from  both training and test datasets are used to train and test the model respectively. The test dataset subset also act as the reference data for data drift monitoring. Whenever unseen samples are needed, data is randomly fetched from the training dataset.
</p>

<h3>Model Experimentation and Tracking</h3>
<!-- <img title="Model Experimentation and Tracking" src="src\image\model exp.png"> -->
<p>
I first started experimenting the solution on a Jupyter Notebook, <i>starter.ipynb</i>. Then refactored the notebook codes into <i>training/training.py</i>. Amazon EC2, Amazon RDS and Amazon S3 are setup to host the MLFlow tracking server, to store MLFlow metadata and artifacts respectively.

Once an Amazon EC2 instance is running with MLFlow installations done, run this command with all the variables substituted:
```
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://postgres:<password>@<aws_rds_hostname>:5432/mlflow_db --default-artifact-root s3://<s3_bucket_name>
```
The MLFlow UI will be available at `http://<ec2_public_address>:5000`
<img title="MLFlow UI" src="src\images\mlflow_ui.png">

Once the MLFlow tracking server is ready, the training code can be ran. The overview of <i>training/training.py</i> is as such:
<ol>
  <li>Initialize MLFlow tracking URI and experiment name</li>
  <li>Get the training and test datasets</li>
  <li>Data preprocessing</li>
  <li>Model training and hyperparameter runing</li>
  <li>Model registry staging</li>
</ol>

To ease the modelling process, it is deployed to Prefect Cloud.
For Prefect Version 2.19.9, these are the steps to create a deployment:
<ol>
  <li>Make sure terminal directory is at root.</li>
  <li>Run <i>prefect init</i> and follow the UI instructions, I chose local</li>
  <li>Run <i>prefect deploy</i>  and follow the UI instructions</li>
  <li>Run <i>prefect worker start --pool 'mlops-capstone'</i> to start worker</li>
</ol>
To use prefect withot deployment:
<ol>
  <li>Make sure terminal directory is at root.</li>
  <li>Run <i>prefect server start</i> and prefect server will start running at port:4200</li>
  <li>Run <i>python training/training.py</i> to see the workflow orchestration</li>
</ol>
Once the deployment is successful, a flow run can be executed from `Spam Detector Capstone/mlops-capstone-spam-detector`. <img title="Prefect Training Flow" src="src\images\prefect_orch.png">

At the end of the run, a model will be staged in MLFlow 
Model Registry.
<img title="MLFlow registry" src="src\images\mlflow_model.png">
</p>

<h3>Model Deployment</h3>
<!-- <img title="Model Deployment" src="src\image\deployment.png"> -->
<p>
In this section, I created an prediction script, <i>predict.py</i> with the following steps:
<ol>
  <li>Get unseen data, simulated by getting randomly from the training dataset</li>
  <li>Preprocess unseen data</li>
  <li>Initialize MLFlow tracking URI</li>
  <li>Load production model from MLFlow Model Registry</li>
  <li>Predict on the unseen data</li>
  <li>Write results to parquet, to simulate passing the results to email client API</li>
</ol>

The prediction script is containerized using a DockerFile with can be hosted on Amazon ECS.

Steps to build and run the container:
<ol>
  <li>Make sure terminal directory is at root.</li>
  <li>Run <i>docker build -t spam-detection-predict:v1 -f .\deployment\Dockerfile .</i></li>
  <li>Run <i>docker run -it -e AWS_ACCESS_KEY_ID=  &lt;XXX&gt; -e AWS_SECRET_ACCESS_KEY= &lt;XXX&gt; spam-detection-predict:v1  &lt;MLFLOW_TRACKING_URL&gt;</i> with all the variables substituted. 
</li>
</ol>

Once the run the successful, you should see a <i>spam_detection.parquet</i> file in the S3 bucket with the current year and date as part of the prefix key.
<img title="Spam Prediction in S3" src="src\images\s3_deployed.png">
</p>

<h3>Monitoring</h3>
<!-- <img title="Monitoring" src="src\image\monitoring.png"> -->
<p>
The monitoring section leverages the <a href='https://www.evidentlyai.com/' target="_blank">Evidently AI</a> library. I have mainly modified  <i>evidently_metrics_calculation.py</i> from the course to fit my use case.
The steps of the script are:
<ol>
  <li>Prepare PostgreSQL database and table</li>
  <li>Initialize MLFlow tracking URI</li>
  <li>Load production model from MLFlow Model Registry</li>
  <li>Get unseen data, simulated by getting randomly from the training dataset</li>
  <li>Preprocess unseen data</li>
  <li>Predict on the unseen data</li>
  <li>Get reference data from MLFlow production model run</li>
  <li>Calculate drift metrics at 5 random time intervals</li>
</ol>

From the root directory, run
```
docker compose -f .\monitoring\docker-compose.yml up --build
```
to prepare PostgreSQL, Adminer and Grafana. This can be deployed to Amazon ECS.

Since the text embeddings play a large role in the model's performance, three embedding drift metrics are used, namely with the methods: classifier model, maximum mean discrepancy and cosine distance. This is based on a good <a href="https://www.evidentlyai.com/blog/embedding-drift-detection"  target="_blank">blog writeup</a> by Evidently AI. A drift is considered to be occured when at least two drifts from the mentioned three methods are detected. When a drift is detected, a flow from the model training deployment `Spam Detector Capstone/mlops-capstone-spam-detector` is automatically ran to retrain the model.

The monitoring script has also been deployed to Prefect Cloud with the same steps as described earlier. A flow run can be executed by running <i>python monitoring/evidently_metrics_calculation.py</i>.
<img title="Prefect Monitoring Flow" src="src\images\batch_monitoring.png">

Once everything has run successfully, Adminer can be logged in from `http://localhost:8081/`
<img title="Adminer Settings" src="src\images\adminer_settings.png">
and you should see some data in the `embedding_drift_metrics` table.
<img title="Adminer Data" src="src\images\adminer_table.png">

The Grafana dashboard can be accessed from `http://localhost:3001/`.
<img title="Grafana Dashboard" src="src\images\grafana_vizualization.png">

</p>

<h3>Code best practices</h3>
<p>
The following have been developed:
<ol>
  <li>Unit test</li>
  <li>Integration test</li>
  <li>auto code formatter using Black</li>
  <li>Makefile</li>
</ol>
Unit test and Integration test can be ran using Makefile by running <i> make unit_and_integration_tests</i>
<img title="Makefile Test" src="src\images\makefile_test.png">

</p>
