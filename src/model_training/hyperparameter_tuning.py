"""
Hyperparameter Tuning Class for Task 5
Handles Grid Search and Random Search for model optimization
"""
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

class HyperparameterTuning:
    """Class for hyperparameter tuning"""
    
    def __init__(self, config):
        self.config = config
        self.tuned_models = {}
        self.best_params = {}
        self.best_scores = {}
    
    def tune_model(self, model, model_name, X_train, y_train, 
                   tuning_method='grid', cv=None, n_iter=None):
        """Tune hyperparameters for a single model"""
        print(f"🎯 Tuning {model_name} using {tuning_method} search...")
        
        if model_name not in self.config.HYPERPARAM_GRIDS:
            print(f"   ⚠️  No hyperparameter grid defined for {model_name}")
            return model
        
        param_grid = self.config.HYPERPARAM_GRIDS[model_name]
        
        # Set CV folds
        if cv is None:
            cv = self.config.CV_FOLDS
        
        # Set n_iter for random search
        if n_iter is None:
            n_iter = self.config.N_ITER_RANDOM_SEARCH
        
        if tuning_method.lower() == 'grid':
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1,
                return_train_score=True
            )
        elif tuning_method.lower() == 'random':
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='roc_auc',
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
                verbose=1,
                return_train_score=True
            )
        else:
            raise ValueError(f"Unknown tuning method: {tuning_method}")
        
        # Perform search
        search.fit(X_train, y_train)
        
        # Store results
        self.tuned_models[model_name] = search.best_estimator_
        self.best_params[model_name] = search.best_params_
        self.best_scores[model_name] = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'cv_results': search.cv_results_
        }
        
        print(f"   ✅ Best score: {search.best_score_:.4f}")
        print(f"   ✅ Best parameters: {search.best_params_}")
        
        return search.best_estimator_
    
    def tune_all_models(self, models, X_train, y_train, tuning_method='grid'):
        """Tune hyperparameters for all models"""
        print("\n🔧 Tuning hyperparameters for all models...")
        
        for model_name, model in models.items():
            try:
                tuned_model = self.tune_model(
                    model=model,
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    tuning_method=tuning_method
                )
                self.tuned_models[model_name] = tuned_model
            except Exception as e:
                print(f"   ❌ Failed to tune {model_name}: {str(e)}")
                continue
        
        print(f"\n✅ Tuned {len(self.tuned_models)} models")
        return self.tuned_models
    
    def get_tuning_results(self, model_name=None):
        """Get tuning results for specific model or all models"""
        if model_name:
            return {
                'best_model': self.tuned_models.get(model_name),
                'best_params': self.best_params.get(model_name),
                'best_score': self.best_scores.get(model_name, {}).get('best_score')
            }
        else:
            return {
                'tuned_models': self.tuned_models,
                'best_params': self.best_params,
                'best_scores': self.best_scores
            }
    
    def compare_tuning_results(self):
        """Compare tuning results across models"""
        if not self.best_scores:
            return "No tuning results available"
        
        comparison = []
        for model_name, scores in self.best_scores.items():
            comparison.append({
                'model': model_name,
                'best_score': scores['best_score'],
                'n_params_tried': len(scores['cv_results']['params'])
            })
        
        return sorted(comparison, key=lambda x: x['best_score'], reverse=True)