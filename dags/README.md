# Orchestration
The assignment permits a clearly structured scheduled Python workflow. `run_pipeline.py` is the orchestration entry point and calls each stage in dependency order. Airflow/Prefect are not required for this implementation. A Windows Task Scheduler job can call `python D:\\path\\to\\recommendation-pipeline\\run_pipeline.py` if scheduled execution is desired.
