# core/segment/classification_model_test.py
import os
import pandas as pd # type: ignore
import numpy as np # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import seaborn as sns # type: ignore
from typing import Dict,  Optional, Any
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import cross_val_score # type: ignore
from sklearn.utils.class_weight import compute_class_weight # type: ignore  # noqa: F401
from src.utils import get_path
import warnings
warnings.filterwarnings('ignore')



class ClassificationModelTest:
    """
    Supervised Learning for Perk Assignment.
    This class uses supervised learning to predict the best perk for users.
    It supports both Support Vector Machines (SVM) and Random Forest classifiers.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = 'assigned_perk',
        test_size: float = 0.3,
        random_state: int = 42,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the ClassificationModelTest.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing user data and assigned perks.
        target_col : str
            Name of the target column (default: 'assigned_perk').
        test_size : float
            Proportion of data to use for testing (default: 0.3).
        random_state : int
            Random seed for reproducibility (default: 42).
        output_dir : str, optional
            Directory for saving outputs.
        """
        self.data = data.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.output_dir = output_dir or os.path.join(get_path('reports'), 'segment', 'classifier')
        self.data_path = get_path('classifier')

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)

        # Define available perks
        self.all_perks = [
            "1 night free hotel plus flight",
            "free hotel meal",
            "free checked bags",
            "no cancellation fees",
            "exclusive discounts"
        ]

        # Remove unwanted columns if they exist
        self._remove_unwanted_columns()

        # Initialize variables
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.preprocessor = None
        self.models = {}
        self.metrics = {}
        self.visualizations = {}

        print("✅ ClassificationModelTest initialized")
        print(f"   Data shape: {self.data.shape}")
        print(f"   Target column: {self.target_col}")

    def _remove_unwanted_columns(self):
        """
        Remove unwanted columns from the data.
        """
        unwanted_columns = ['cluster', 'segment_name', 'persona_type', 'home_city', 'assigned_group']
        for col in unwanted_columns:
            if col in self.data.columns:
                self.data.drop(columns=[col], inplace=True)
        print(f"   ✅ Removed unwanted columns: {unwanted_columns}")

    def _preprocess_data(self):
        """
        Preprocess the data: split into features and target, and handle missing values.
        """
        # Drop rows where target is missing
        self.data.dropna(subset=[self.target_col], inplace=True)

        # Separate features and target
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]

        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Identify numeric, categorical, and boolean columns
        numeric_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        boolean_cols = self.X_train.select_dtypes(include=['bool']).columns.tolist()

        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        boolean_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(drop='if_binary'))
        ])

        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols),
                ('bool', boolean_transformer, boolean_cols)
            ]
        )

        print(f"   ✅ Data preprocessed")  # noqa: F541
        print(f"      Train shape: {self.X_train.shape}")
        print(f"      Test shape: {self.X_test.shape}")

    def train_models(self):
        """
        Train SVM and Random Forest models.
        """
        if self.X_train is None:
            self._preprocess_data()

        # Define models
        svm_model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=self.random_state))
        ])

        rf_model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=self.random_state))
        ])

        # Train models
        svm_model.fit(self.X_train, self.y_train)
        rf_model.fit(self.X_train, self.y_train)

        self.models = {
            'svm': svm_model,
            'random_forest': rf_model
        }

        print(f"   ✅ Models trained")  # noqa: F541

    def evaluate_models(self):
        """
        Evaluate models using various metrics.
        """
        if not self.models:
            self.train_models()

        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test) if hasattr(model.named_steps['classifier'], 'predict_proba') else None

            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(self.y_test, y_proba, multi_class='ovr') if y_proba is not None else None

            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')

            # Store metrics
            self.metrics[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }

            print(f"   ✅ {model_name.capitalize()} evaluated")
            print(f"      Accuracy: {accuracy:.3f}")
            print(f"      Precision: {precision:.3f}")
            print(f"      Recall: {recall:.3f}")
            print(f"      F1: {f1:.3f}")
            if roc_auc is not None:
                print(f"      ROC AUC: {roc_auc:.3f}")
            print(f"      CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    def visualize_results(self):
        """
        Visualize model results.
        """
        if not self.metrics:
            self.evaluate_models()

        # Confusion Matrices
        for model_name, metrics in self.metrics.items():
            cm = metrics['confusion_matrix']
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.all_perks, yticklabels=self.all_perks)
            plt.title(f'Confusion Matrix - {model_name.capitalize()}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            save_path = os.path.join(self.output_dir, f'confusion_matrix_{model_name}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.show()
            plt.close()
            self.visualizations[f'confusion_matrix_{model_name}'] = save_path
            print(f"   ✅ Saved confusion matrix for {model_name}: {save_path}")

        # Feature Importance for Random Forest
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']
            feature_names = (rf_model.named_steps['preprocessor']
                             .get_feature_names_out())
            importances = rf_model.named_steps['classifier'].feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 8))
            plt.title('Feature Importances - Random Forest')
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, 'feature_importances.png')
            plt.savefig(save_path, dpi=300)
            plt.show()
            plt.close()
            self.visualizations['feature_importances'] = save_path
            print(f"   ✅ Saved feature importances: {save_path}")

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of the classification model test.

        Returns
        -------
        Dict[str, Any]
            Comprehensive report of the classification model test.
        """
        if not self.metrics:
            self.evaluate_models()

        report = {
            'metadata': {
                'test_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_shape': self.data.shape,
                'train_shape': self.X_train.shape,
                'test_shape': self.X_test.shape,
                'target_column': self.target_col,
                'random_state': self.random_state
            },
            'metrics': self.metrics,
            'visualizations': self.visualizations
        }

        report_path = os.path.join(self.data_path, 'classification_report.json')
        pd.DataFrame(report).to_json(report_path, indent=4)
        print(f"   ✅ Saved classification report: {report_path}")

        return report

    def export_results(self, filename: str = 'classification_results.csv') -> str:
        """
        Export test results to CSV.

        Parameters
        ----------
        filename : str
            Name of the output file.

        Returns
        -------
        str
            Path to saved file.
        """
        if self.X_test is None:
            self._preprocess_data()

        # Predictions for each model
        for model_name, model in self.models.items():
            self.X_test[f'{model_name}_pred'] = model.predict(self.X_test)

        # Save the test data with predictions
        save_path = os.path.join(self.data_path, filename)
        self.X_test.to_csv(save_path, index=False)
        print(f"   ✅ Saved test results with predictions: {save_path}")

        return save_path

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run complete analysis: preprocessing, training, evaluation, and visualization.

        Returns
        -------
        Dict[str, Any]
            Complete analysis results.
        """
        print("\n🚀 Running complete supervised learning analysis...\n")

        # Step 1: Preprocess data
        print("[STEP 1] Preprocessing data...")
        self._preprocess_data()
        print("   ✅ Data preprocessing finished")
        print(f"   Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")

        # Step 2: Train models
        print("\n[STEP 2] Training models (SVM & Random Forest)...")
        self.train_models()
        print("   ✅ Model training completed")
        print(f"   Models available: {list(self.models.keys())}")

        # Step 3: Evaluate models
        print("\n[STEP 3] Evaluating models...")
        self.evaluate_models()
        print("   ✅ Model evaluation completed")
        for model_name, metrics in self.metrics.items():
            print(f"   {model_name.capitalize()} -> Accuracy: {metrics['accuracy']:.3f}, "
                f"F1: {metrics['f1']:.3f}, CV Mean: {metrics['cv_mean']:.3f}")

        # Step 4: Visualize results
        print("\n[STEP 4] Generating visualizations...")
        self.visualize_results()
        print("   ✅ Visualizations saved")
        print(f"   Files: {list(self.visualizations.values())}")

        # Step 5: Generate report
        print("\n[STEP 5] Compiling report...")
        report = self.generate_report()
        print("   ✅ Report generated")
        print(f"   Report keys: {list(report.keys())}")

        # Step 6: Export results
        print("\n[STEP 6] Exporting results to CSV...")
        self.export_results()
        print("   ✅ Results exported")

        print("\n✅ Supervised learning analysis completed successfully!")
        return report

