import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class BatchPlotter:
    """
    IEEE-Grade Visualization Engine for Batch and Comparative Analysis.
    Focuses on generating high-fidelity statistical figures suitable for 
    research publications across Dataframe metric aggregations.
    """

    @staticmethod
    def _ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)
        
    @staticmethod
    def set_publication_style():
        """Configures matplotlib for IEEE paper standards."""
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'axes.grid': True,
            'grid.alpha': 0.3
        })

    @staticmethod
    def render_stability_boxplots(df: pd.DataFrame, save_dir: str):
        """
        Generates comparative boxplots for stability metrics across different Hostility Levels or Modes.
        Requires 'hostility_level' or 'mode' as a categorical column in the dataframe.
        """
        BatchPlotter._ensure_dir(save_dir)
        BatchPlotter.set_publication_style()

        metrics_of_interest = ['replan_frequency', 'collision_rate', 'path_stability_index']
        category_col = 'hostility_level' if 'hostility_level' in df.columns else 'mode'
        
        if category_col not in df.columns:
            return # Cannot group

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics_of_interest):
            if metric in df.columns:
                sns.boxplot(x=category_col, y=metric, data=df, ax=axes[idx], palette="muted", showmeans=True)
                axes[idx].set_title(f"Distribution of {metric.replace('_', ' ').title()}")
                axes[idx].set_ylabel(metric.replace('_', ' ').title())
                axes[idx].set_xlabel("Scenario Configure")
                
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "stability_distributions.pdf"), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, "stability_distributions.png"), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def render_semantic_correlation_heatmap(df: pd.DataFrame, save_dir: str):
        """
        Generates a correlation heatmap between Semantic routing metrics and physical constraints.
        """
        BatchPlotter._ensure_dir(save_dir)
        BatchPlotter.set_publication_style()
        
        cols = ['priority_satisfaction_percent', 'semantic_purity_index', 
                'visited', 'final_battery', 'replans', 'path_stability_index']
        
        # Filter columns that actually exist in the dataframe
        valid_cols = [c for c in cols if c in df.columns]
        
        if len(valid_cols) < 2:
            return
            
        corr_matrix = df[valid_cols].astype(float).corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
        plt.title("Correlation Matrix of Semantic & Physical Metrics")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "metric_correlations.pdf"), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, "metric_correlations.png"), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def render_efficiency_pareto(df: pd.DataFrame, save_dir: str):
        """
        Plots a scatter Pareto distribution analyzing Priority Satisfaction versus Energy Consumed.
        """
        BatchPlotter._ensure_dir(save_dir)
        BatchPlotter.set_publication_style()
        
        if 'priority_satisfaction_percent' not in df.columns or 'final_battery' not in df.columns:
            return
            
        plt.figure(figsize=(8, 6))
        
        # We plot Energy Consumed (derived) vs Satisfaction
        # Assuming starting battery is constant across batch. If not, this is roughly inverse to final battery.
        max_bat = df['final_battery'].max() * 1.5 # rough estimate of starting bat if not logged
        energy_used = max_bat - df['final_battery']
        
        sns.scatterplot(
            x=energy_used, 
            y=df['priority_satisfaction_percent'], 
            hue=df.get('mode', None),
            palette="viridis",
            s=100, 
            alpha=0.8
        )
        
        plt.title("Pareto Efficiency: Priority Execution vs. Energy Cost")
        plt.xlabel("Estimated Energy Consumed (Joules)")
        plt.ylabel("Priority Satisfaction (%)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "efficiency_pareto.pdf"), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, "efficiency_pareto.png"), dpi=300, bbox_inches='tight')
        plt.close()
