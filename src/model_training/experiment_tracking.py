"""
Experiment Tracking Class for Task 5
Handles MLflow integration for experiment tracking and model registry
"""
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
from datetime import datetime

class ExperimentTracking:
    """Class for MLflow experiment tracking"""
    
    def __init__(self, config):
        self.config = config
        self.experiment_id = None
        self.runs = {}
        
        # Setup MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.config.MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                name=self.config.MLFLOW_EXPERIMENT_NAME
            )
        else:
            self.experiment_id = experiment.experiment_id
        
        print(f"🔬 MLflow Experiment: {self.config.MLFLOW_EXPERIMENT_NAME}")
        print(f"   Tracking URI: {self.config.MLFLOW_TRACKING_URI}")
        print(f"   Experiment ID: {self.experiment_id}")
    
    def start_run(self, run_name=None, nested=False):
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            nested=nested
        )
        
        self.runs[run_name] = run.info.run_id
        print(f"   Started run: {run_name} (ID: {run.info.run_id})")
        
        return run
    
    def log_params(self, params):
        """Log parameters to current run"""
        if isinstance(params, dict):
            mlflow.log_params(params)
        else:
            print("⚠️  Parameters must be a dictionary")
    
    def log_metrics(self, metrics):
        """Log metrics to current run"""
        if isinstance(metrics, dict):
            mlflow.log_metrics(metrics)
        else:
            print("⚠️  Metrics must be a dictionary")
    
    def log_model(self, model, model_name, signature=None, input_example=None):
        """Log model to MLflow"""
        if signature is None and input_example is not None:
            signature = infer_signature(input_example, model.predict(input_example))
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            signature=signature,
            input_example=input_example,
            registered_model_name=f"credit_risk_{model_name}"
        )
        
        print(f"   Logged model: {model_name}")
    
    def log_artifact(self, local_path, artifact_path=None):
        """Log artifact to MLflow"""
        mlflow.log_artifact(local_path, artifact_path)
        print(f"   Logged artifact: {local_path}")
    
    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()
        print("   Ended run")
    
    def register_model(self, model_name, run_id=None, model_path="model"):
        """Register model in MLflow Model Registry"""
        if run_id is None:
            # Get current run ID
            run_id = mlflow.active_run().info.run_id
        
        model_uri = f"runs:/{run_id}/{model_path}"
        
        try:
            mlflow.register_model(
                model_uri=model_uri,
                name=f"credit_risk_{model_name}"
            )
            print(f"✅ Registered model: credit_risk_{model_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to register model: {str(e)}")
            return False
    
    def get_experiment_runs(self):
        """Get all runs from current experiment"""
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        return runs
    
    def compare_runs(self, metric='metrics.roc_auc'):
        """Compare runs based on a metric"""
        runs_df = self.get_experiment_runs()
        
        if runs_df.empty:
            return "No runs found"
        
        # Sort by metric
        if metric in runs_df.columns:
            sorted_runs = runs_df.sort_values(by=metric, ascending=False)
            return sorted_runs[['run_id', 'tags.mlflow.runName', metric]].head(10)
        else:
            return f"Metric {metric} not found in runs"