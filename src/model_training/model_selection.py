"""
Model Selection and Training Class for Task 5
Handles model initialization, training, and basic evaluation
"""
import importlib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelSelection:
    """Class for model selection and training"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.trained_models = {}
        
    def initialize_models(self, model_names=None):
        """Initialize models from configuration"""
        print("🤖 Initializing models...")
        
        if model_names is None:
            model_names = list(self.config.MODELS.keys())
        
        for model_name in model_names:
            if model_name in self.config.MODELS:
                model_config = self.config.MODELS[model_name]
                
                # Dynamically import the model class
                module_name, class_name = model_config["class"].rsplit(".", 1)
                module = __import__(module_name, fromlist=[class_name])
                ModelClass = getattr(module, class_name)
                
                # Create model instance with default parameters
                model = ModelClass(**model_config["default_params"])
                self.models[model_name] = model
                
                print(f"   ✅ {model_name}: {type(model).__name__}")
            else:
                print(f"   ⚠️  {model_name} not found in configuration")
        
        return self.models
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train all initialized models"""
        print("\n🚀 Training models...")
        
        for model_name, model in self.models.items():
            print(f"\n   Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                self.trained_models[model_name] = model
                
                # Evaluate on training data
                train_pred = model.predict(X_train)
                train_pred_proba = model.predict_proba(X_train)[:, 1]
                
                train_metrics = self._calculate_metrics(y_train, train_pred, train_pred_proba)
                
                print(f"     Training metrics:")
                print(f"       Accuracy:  {train_metrics['accuracy']:.4f}")
                print(f"       Precision: {train_metrics['precision']:.4f}")
                print(f"       Recall:    {train_metrics['recall']:.4f}")
                print(f"       F1-Score:  {train_metrics['f1']:.4f}")
                print(f"       ROC-AUC:   {train_metrics['roc_auc']:.4f}")
                
                # Evaluate on validation data if provided
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    val_pred_proba = model.predict_proba(X_val)[:, 1]
                    
                    val_metrics = self._calculate_metrics(y_val, val_pred, val_pred_proba)
                    
                    print(f"     Validation metrics:")
                    print(f"       Accuracy:  {val_metrics['accuracy']:.4f}")
                    print(f"       Precision: {val_metrics['precision']:.4f}")
                    print(f"       Recall:    {val_metrics['recall']:.4f}")
                    print(f"       F1-Score:  {val_metrics['f1']:.4f}")
                    print(f"       ROC-AUC:   {val_metrics['roc_auc']:.4f}")
                
            except Exception as e:
                print(f"     ❌ Failed to train {model_name}: {str(e)}")
                continue
        
        print(f"\n✅ Trained {len(self.trained_models)} models successfully")
        return self.trained_models
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate evaluation metrics"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def get_model(self, model_name):
        """Get a specific trained model"""
        return self.trained_models.get(model_name)
    
    def get_all_models(self):
        """Get all trained models"""
        return self.trained_models