"""
LaTeX Table Exporter
====================
Converts aggregated simulation results (JSON) into publication-ready
LaTeX table fragments for IEEE conference papers.

Usage (library):
    from metrics.latex_exporter import LatexExporter
    LatexExporter.from_json("results/aggregated/batch_summary.json", "output.tex")

Usage (CLI):
    python -m metrics.latex_exporter results/aggregated/batch_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


class LatexExporter:
    """Generate LaTeX tabular environments from metric dictionaries."""

    # ── From JSON file ───────────────────────────────────────────────

    @staticmethod
    def from_json(json_path: str, output_path: str | None = None) -> str:
        """Load a batch_summary.json and convert to LaTeX."""
        with open(json_path, "r") as fh:
            data = json.load(fh)
        return LatexExporter.render(data, output_path)

    # ── Core rendering ───────────────────────────────────────────────

    @staticmethod
    def render(metrics: Dict[str, dict], output_path: str | None = None) -> str:
        """
        Render a metric dict of the form
            { "metric_name": {"mean": float, "std": float, "ci95": float}, ... }
        into a LaTeX table.

        Returns the LaTeX string and optionally writes to *output_path*.
        """
        header = (
            r"\begin{table}[htbp]" "\n"
            r"\centering" "\n"
            r"\caption{Aggregated Simulation Metrics}" "\n"
            r"\label{tab:sim_metrics}" "\n"
            r"\begin{tabular}{lccc}" "\n"
            r"\toprule" "\n"
            r"Metric & Mean & Std & CI$_{95\%}$ \\" "\n"
            r"\midrule" "\n"
        )

        rows: List[str] = []
        for name, vals in metrics.items():
            label = LatexExporter._format_label(name)
            mean = vals.get("mean", 0.0)
            std = vals.get("std", 0.0)
            ci = vals.get("ci95", 0.0)
            rows.append(f"{label} & {mean:.4f} & {std:.4f} & $\\pm${ci:.4f} \\\\")

        footer = (
            r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table}"
        )

        body = "\n".join(rows) + "\n"
        latex = header + body + footer

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as fh:
                fh.write(latex)
            print(f"[LatexExporter] Written → {output_path}")

        return latex

    # ── Ablation table ───────────────────────────────────────────────

    @staticmethod
    def render_ablation(ablation_results: Dict[str, dict], output_path: str | None = None) -> str:
        """
        Render ablation delta table:
            Factor | Metric | Δ (%)
        """
        header = (
            r"\begin{table}[htbp]" "\n"
            r"\centering" "\n"
            r"\caption{Ablation Study — Relative Change (\%)}" "\n"
            r"\label{tab:ablation}" "\n"
            r"\begin{tabular}{llr}" "\n"
            r"\toprule" "\n"
            r"Factor & Metric & $\Delta$ (\%) \\" "\n"
            r"\midrule" "\n"
        )

        rows: List[str] = []
        for factor, data in ablation_results.items():
            delta = data.get("delta", {})
            for metric, pct in delta.items():
                label = LatexExporter._format_label(factor)
                m_label = LatexExporter._format_label(metric)
                rows.append(f"{label} & {m_label} & {pct:+.2f} \\\\")

        footer = (
            r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table}"
        )

        body = "\n".join(rows) + "\n"
        latex = header + body + footer

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as fh:
                fh.write(latex)
            print(f"[LatexExporter] Written → {output_path}")

        return latex

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _format_label(snake_name: str) -> str:
        """Convert ``snake_case`` to Title Case with underscores replaced by spaces."""
        return snake_name.replace("_", " ").title()


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LaTeX Table Exporter")
    parser.add_argument("input", help="Path to JSON metric file")
    parser.add_argument("-o", "--output", default=None, help="Output .tex path")
    args = parser.parse_args()

    out = args.output or args.input.replace(".json", ".tex")
    LatexExporter.from_json(args.input, out)


if __name__ == "__main__":
    main()
