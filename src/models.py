# src/models.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    balanced_accuracy_score, f1_score
)
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to balance classes"""
    smote = SMOTE(random_state=random_state, k_neighbors=3)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    return X_balanced, y_balanced, smote

def train_random_forest(X_train, y_train, n_estimators=300, random_state=42, 
                       max_depth=15, class_weight='balanced'):
    """Train an improved Random Forest Classifier with class balancing"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train, n_estimators=200, random_state=42):
    """Train a Gradient Boosting Classifier"""
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=7,
        min_samples_split=10,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, show_plot=True):
    """Evaluate the model with comprehensive metrics and plot confusion matrix"""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print("="*60)
    print("MODEL EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred))
    
    if show_plot:
        # Plot confusion matrices
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        axes[0].set_title("Confusion Matrix (Counts)")
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1])
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
        axes[1].set_title("Confusion Matrix (Normalized)")
        
        plt.tight_layout()
        plt.show()
    
    return y_pred, {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
