from kfp.client import Client

print("Connecting to Kubeflow Backend...")
# Connect directly to the port-forwarded API (Bypassing UI bugs)
client = Client(host='http://localhost:8080')

print("Submitting Pipeline directly to Engine...")
try:
    run = client.create_run_from_pipeline_package(
        pipeline_file='fraud_pipeline.yaml',
        arguments={},
        run_name='Direct_API_Run',
        experiment_name='Terminal_Experiment'
    )
    print(f"SUCCESS! Engine accepted the pipeline.")
    print(f"Run ID: {run.run_id}")
    print("Go check the UI 'Runs' tab now!")
except Exception as e:
    print(f"Backend Error: {e}")