# make all MLFLOW_TRACKING_URL=<value>

unit_and_integration_tests:
	bash test/run.sh http://16.170.204.58:5000/

quality_checks:
	black .