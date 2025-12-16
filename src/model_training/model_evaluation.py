"""
Model Evaluation Class for Task 5
Handles model evaluation, metric calculation, and visualization
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report,
    average_precision_score
)
import joblib
import json

class ModelEvaluation:
    """Class for comprehensive model evaluation"""
    
    def __init__(self, config):
        self.config = config
        self.evaluation_results = {}
        
    def evaluate_model(self, model, model_name, X_test, y_test):
        """Comprehensive evaluation of a single model"""
        print(f"📊 Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_all_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.evaluation_results[model_name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist()
        }
        
        # Print summary
        self._print_evaluation_summary(model_name, metrics, cm)
        
        return self.evaluation_results[model_name]
    
    def _calculate_all_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate all evaluation metrics"""
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else 0
        }
        
        # Advanced metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics.update({
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False Positive Rate
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,  # False Negative Rate
            'prevalence': (tp + fn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        })
        
        # For imbalanced data
        if y_pred_proba is not None:
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
            
            # Calculate at different thresholds
            thresholds = [0.3, 0.5, 0.7]
            for threshold in thresholds:
                y_pred_thresh = (y_pred_proba >= threshold).astype(int)
                metrics[f'precision_at_{threshold}'] = precision_score(y_true, y_pred_thresh, zero_division=0)
                metrics[f'recall_at_{threshold}'] = recall_score(y_true, y_pred_thresh, zero_division=0)
        
        return metrics
    
    def _print_evaluation_summary(self, model_name, metrics, cm):
        """Print evaluation summary"""
        print(f"   Model: {model_name}")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        if 'average_precision' in metrics:
            print(f"   Avg Precision: {metrics['average_precision']:.4f}")
        
        print(f"\n   Confusion Matrix:")
        print(f"               Predicted")
        print(f"               0       1")
        print(f"   Actual 0   {cm[0,0]:<6}  {cm[0,1]:<6}")
        print(f"           1   {cm[1,0]:<6}  {cm[1,1]:<6}")
        
        print(f"\n   Additional Metrics:")
        print(f"   Specificity: {metrics['specificity']:.4f}")
        print(f"   NPV:         {metrics['npv']:.4f}")
        print(f"   FPR:         {metrics['fpr']:.4f}")
        print(f"   FNR:         {metrics['fnr']:.4f}")
    
    def evaluate_all_models(self, models, X_test, y_test):
        """Evaluate all models"""
        print("\n" + "="*60)
        print("📈 COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        for model_name, model in models.items():
            try:
                self.evaluate_model(model, model_name, X_test, y_test)
            except Exception as e:
                print(f"❌ Failed to evaluate {model_name}: {str(e)}")
                continue
        
        return self.evaluation_results
    
    def compare_models(self):
        """Compare all evaluated models"""
        if not self.evaluation_results:
            return "No evaluation results available"
        
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'TP': metrics['true_positives'],
                'FP': metrics['false_positives']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Find best model based on ROC-AUC
        best_model_idx = comparison_df['ROC-AUC'].astype(float).idxmax()
        best_model = comparison_df.loc[best_model_idx, 'Model']
        best_roc_auc = comparison_df.loc[best_model_idx, 'ROC-AUC']
        
        print("\n🏆 MODEL COMPARISON RESULTS:")
        print("="*60)
        print(comparison_df.to_string(index=False))
        print(f"\n🏆 Best Model: {best_model} (ROC-AUC: {best_roc_auc})")
        
        return {
            'comparison_df': comparison_df,
            'best_model': best_model,
            'best_model_results': self.evaluation_results.get(best_model)
        }
    
    def create_evaluation_plots(self, model_name, X_test, y_test, save_path=None):
        """Create evaluation plots for a model"""
        if model_name not in self.evaluation_results:
            print(f"Model {model_name} not evaluated yet")
            return
        
        results = self.evaluation_results[model_name]
        y_pred = results['predictions']
        y_pred_proba = results['probabilities']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Evaluation - {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix Heatmap
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = results['metrics']['roc_auc']
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = results['metrics'].get('average_precision', 0)
        
        axes[1, 0].plot(recall, precision, color='green', lw=2,
                       label=f'Avg Precision = {avg_precision:.3f}')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend(loc="lower left")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature Importance (if available)
        model = results['model']
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            if feature_importance is not None:
                # Get feature names (assuming X_test is DataFrame)
                if hasattr(X_test, 'columns'):
                    feature_names = X_test.columns.tolist()
                else:
                    feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names[:len(feature_importance)],
                    'importance': feature_importance
                }).sort_values('importance', ascending=False).head(15)
                
                axes[1, 1].barh(range(len(importance_df)), importance_df['importance'])
                axes[1, 1].set_yticks(range(len(importance_df)))
                axes[1, 1].set_yticklabels(importance_df['feature'])
                axes[1, 1].set_xlabel('Importance')
                axes[1, 1].set_title('Top 15 Feature Importances')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Plot saved: {save_path}")
        
        plt.show()
        return fig
    
    def save_evaluation_results(self, output_dir="reports"):
        """Save evaluation results to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison CSV
        comparison = self.compare_models()
        if isinstance(comparison, dict) and 'comparison_df' in comparison:
            comparison['comparison_df'].to_csv(
                f"{output_dir}/model_comparison.csv", index=False
            )
        
        # Save detailed results JSON
        with open(f"{output_dir}/evaluation_results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for model_name, results in self.evaluation_results.items():
                json_results[model_name] = {
                    'metrics': results['metrics'],
                    'classification_report': results['classification_report'],
                    'confusion_matrix': results['confusion_matrix']
                }
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Evaluation results saved to {output_dir}/")