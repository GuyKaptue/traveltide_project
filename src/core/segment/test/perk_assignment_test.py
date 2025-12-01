# test/perk_assignment_test.py

import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

from typing import Dict, Any, Optional, List
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, f1_score, classification_report # type: ignore
from scipy.stats import f_oneway # type: ignore

from sklearn.pipeline import Pipeline # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder # type: ignore

from src.utils import get_path


class PerkAssignmentTest:
    """
    Comprehensive tester for perk assignment strategies:
    - Random baseline
    - Supervised classifier
    - ANOVA significance
    - Visualizations
    """

    def __init__(self, users: pd.DataFrame, manual_seg: pd.DataFrame, ml_seg: pd.DataFrame):
        self.users = users.copy()
        self.manual_seg = manual_seg.copy()
        self.ml_seg = ml_seg.copy()

        # Merge everything on user_id
        self.data = (
            self.users
            .merge(
                self.manual_seg[['user_id', 'assigned_perk', 'persona_type']],
                on='user_id', how='left', suffixes=('', '_manual')
            )
            .merge(
                self.ml_seg[['user_id', 'assigned_perk', 'segment_name', 'cluster']],
                on='user_id', how='left', suffixes=('', '_ml')
            )
            .rename(columns={'assigned_perk': 'assigned_perk_ml'})
        )

        self.output_dir = os.path.join(get_path("reports"), "perk_tests")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✅ PerkAssignmentTest initialized with {len(self.data):,} users")

    # ============================================================
    # 1. Random perk assignment baseline
    # ============================================================
    def random_perk_assignment(self) -> Dict[str, Any]:
        print("[TEST] Random perk assignment baseline...")
        perks = self.manual_seg['assigned_perk'].dropna().unique()
        self.data['random_perk'] = np.random.choice(perks, size=len(self.data))
        manual_agreement = (self.data['random_perk'] == self.data['assigned_perk_manual']).mean()
        ml_agreement = (self.data['random_perk'] == self.data['assigned_perk_ml']).mean()
        print(f"   Agreement with manual: {manual_agreement:.2%}")
        print(f"   Agreement with ML: {ml_agreement:.2%}")
        return {
            "manual_agreement": manual_agreement,
            "ml_agreement": ml_agreement
        }

    # ============================================================
    # 2. Supervised classifier
    # ============================================================
    def supervised_classifier(self, target: str = 'assigned_perk_manual',
                              feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        print(f"[TEST] Supervised classifier predicting {target}...")

        # Drop segmentation/label columns from features
        drop_cols = [
            "assigned_perk_manual", "persona_type", "segment_name_ml",
            "cluster_ml", "assigned_group", "assigned_perk_ml", "random_perk"
        ]
        df = self.data.dropna(subset=[target])
        y = df[target]
        X = df.drop(columns=[col for col in drop_cols if col in df.columns] + [target])

        # Separate numeric and categorical features
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = [col for col in ["gender", "married", "has_children", "home_country", "home_city"]
                                if col in X.columns]

        # Preprocessing pipelines
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        # Full pipeline with classifier
        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
        ])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Fit and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"   Accuracy: {acc:.2%}")
        print(f"   Weighted F1: {f1:.3f}")
        return {
            "accuracy": acc,
            "f1_score": f1,
            "report": classification_report(y_test, y_pred, output_dict=True),
            "classifier": clf,
            "features": numeric_features + categorical_features
        }

    # ============================================================
    # 3. ANOVA feature comparison
    # ============================================================
    def run_anova(self, features: Optional[List[str]] = None) -> Dict[str, Any]:
        print("[TEST] Running ANOVA across perk groups...")
        if features is None:
            features = ["total_spend", "num_trips", "conversion_rate", "RFM_score"]
        results = {}
        for feat in features:
            groups = [
                self.data[self.data['assigned_perk_manual'] == perk][feat].dropna()
                for perk in self.data['assigned_perk_manual'].dropna().unique()
            ]
            if len(groups) > 1:
                f_stat, p_val = f_oneway(*groups)
                results[feat] = {"f_stat": f_stat, "p_value": p_val}
                print(f"   {feat}: F={f_stat:.3f}, p={p_val:.4f}")
        return results

    # ============================================================
    # 4. Visualizations
    # ============================================================
    def plot_anova_significance(self, anova_results: Dict[str, Any]):
        plt.figure(figsize=(8, 6))
        features = list(anova_results.keys())
        pvals = [anova_results[f]['p_value'] for f in features]
        scores = [-np.log10(p) for p in pvals]
        sns.barplot(x=features, y=scores, palette="viridis")
        plt.axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        plt.ylabel("-log10(p-value)")
        plt.title("ANOVA Significance by Feature")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        save_path = os.path.join(self.output_dir, "anova_significance.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Saved ANOVA significance plot: {save_path}")
        return save_path

    def plot_feature_boxplots(self, features: List[str]):
        paths = {}
        for feat in features:
            plt.figure(figsize=(8, 6))
            sns.boxplot(
                data=self.data,
                x="assigned_perk_manual",
                y=feat,
                palette="Set2"
            )
            plt.title(f"{feat} by Perk Group")
            plt.xticks(rotation=45, ha="right")
            save_path = os.path.join(self.output_dir, f"boxplot_{feat}.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            paths[feat] = save_path
            print(f"✅ Saved boxplot for {feat}: {save_path}")
        return paths

    def plot_classifier_importances(self, clf, feature_cols: List[str]):
        try:
            importances = clf.named_steps['classifier'].feature_importances_
        except AttributeError:
            importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(8, 6))
        sns.barplot(
            x=[feature_cols[i] for i in indices],
            y=importances[indices],
            palette="coolwarm"
        )
        plt.title("Classifier Feature Importances")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Importance")
        save_path = os.path.join(self.output_dir, "classifier_importances.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Saved classifier importance plot: {save_path}")
        return save_path

    # ============================================================
    # Master runner
    # ============================================================
    def run_all_tests(self) -> Dict[str, Any]:
        print("\n🚀 Running all perk assignment tests...\n")
        random_results = self.random_perk_assignment()
        clf_perk_results = self.supervised_classifier(target='assigned_perk_manual')
        anova_results = self.run_anova()
        # Generate plots
        anova_plot = self.plot_anova_significance(anova_results)
        boxplots = self.plot_feature_boxplots(list(anova_results.keys()))
        importance_plot = self.plot_classifier_importances(
            clf_perk_results["classifier"], clf_perk_results["features"]
        )
        return {
            "random_baseline": random_results,
            "classifier_perk": clf_perk_results,
            "anova": anova_results,
            "plots": {
                "anova_significance": anova_plot,
                "boxplots": boxplots,
                "classifier_importances": importance_plot
            }
        }
