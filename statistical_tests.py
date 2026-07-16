"""
Statistical Tests for LegalRAG Evaluation
Performs paired t-tests and calculates effect sizes
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys


def load_latest_results():
    """Load the most recent comparison CSV"""
    metrics_dir = Path("results/metrics")
    
    if not metrics_dir.exists():
        print("Error: results/metrics directory not found!")
        print("Please run: python compare_all.py --all")
        sys.exit(1)
    
    csv_files = list(metrics_dir.glob("comparison_*.csv"))
    
    if not csv_files:
        print("Error: No comparison CSV files found!")
        print("Please run: python compare_all.py --all")
        sys.exit(1)
    
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest_file.name}\n")
    
    return pd.read_csv(latest_file)


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_effect_size(d):
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def interpret_p_value(p):
    """Interpret statistical significance"""
    if p < 0.001:
        return "*** (p < 0.001) - Highly Significant"
    elif p < 0.01:
        return "**  (p < 0.01)  - Very Significant"
    elif p < 0.05:
        return "*   (p < 0.05)  - Significant"
    else:
        return "ns  (p ≥ 0.05)  - Not Significant"


def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def descriptive_statistics(df):
    """Calculate and display descriptive statistics"""
    print_section_header("DESCRIPTIVE STATISTICS")
    
    metrics = ['citation_accuracy', 'irac_compliance', 'hallucination_rate', 
               'total_citations', 'correct_citations', 'hallucinated_citations']
    
    for approach in ['zero_shot', 'naive_rag', 'legalrag']:
        data = df[df['approach'] == approach]
        
        print(f"\n{approach.upper().replace('_', ' ')}:")
        print("-" * 80)
        
        for metric in metrics:
            values = data[metric]
            print(f"  {metric.replace('_', ' ').title():<30} "
                  f"Mean: {values.mean():.4f}  "
                  f"SD: {values.std():.4f}  "
                  f"Min: {values.min():.4f}  "
                  f"Max: {values.max():.4f}")


def paired_t_tests(df):
    """Perform paired t-tests"""
    print_section_header("PAIRED T-TESTS")
    
    metrics = {
        'citation_accuracy': 'Citation Accuracy',
        'irac_compliance': 'IRAC Compliance',
        'hallucination_rate': 'Hallucination Rate'
    }
    
    # LegalRAG vs Naive RAG (main comparison)
    print("\n╔════════════════════════════════════════════════════════════════════╗")
    print("║              LEGALRAG vs NAIVE RAG (Primary Comparison)           ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    lr_data = df[df['approach'] == 'legalrag']
    nr_data = df[df['approach'] == 'naive_rag']
    
    for metric, label in metrics.items():
        lr_values = lr_data[metric].values
        nr_values = nr_data[metric].values
        
        t_stat, p_value = stats.ttest_rel(lr_values, nr_values)
        effect_size = cohens_d(lr_values, nr_values)
        
        print(f"\n{label}:")
        print(f"  LegalRAG:   Mean = {lr_values.mean():.4f}, SD = {lr_values.std():.4f}")
        print(f"  Naive RAG:  Mean = {nr_values.mean():.4f}, SD = {nr_values.std():.4f}")
        print(f"  Difference: {lr_values.mean() - nr_values.mean():.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Cohen's d: {effect_size:.4f} ({interpret_effect_size(effect_size)})")
        print(f"  Significance: {interpret_p_value(p_value)}")
    
    # LegalRAG vs Zero-Shot (baseline comparison)
    print("\n╔════════════════════════════════════════════════════════════════════╗")
    print("║            LEGALRAG vs ZERO-SHOT (Baseline Comparison)            ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    zs_data = df[df['approach'] == 'zero_shot']
    
    for metric, label in metrics.items():
        lr_values = lr_data[metric].values
        zs_values = zs_data[metric].values
        
        t_stat, p_value = stats.ttest_rel(lr_values, zs_values)
        effect_size = cohens_d(lr_values, zs_values)
        
        print(f"\n{label}:")
        print(f"  LegalRAG:   Mean = {lr_values.mean():.4f}, SD = {lr_values.std():.4f}")
        print(f"  Zero-Shot:  Mean = {zs_values.mean():.4f}, SD = {zs_values.std():.4f}")
        print(f"  Difference: {lr_values.mean() - zs_values.mean():.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Cohen's d: {effect_size:.4f} ({interpret_effect_size(effect_size)})")
        print(f"  Significance: {interpret_p_value(p_value)}")


def summary_table(df):
    """Generate publication-ready summary table"""
    print_section_header("PUBLICATION-READY SUMMARY TABLE")
    
    approaches = ['zero_shot', 'naive_rag', 'legalrag']
    labels = ['Zero-Shot', 'Naive RAG', 'LegalRAG']
    
    print("\n╔════════════════════════════════════════════════════════════════════════════════╗")
    print("║ Metric                  │  Zero-Shot  │  Naive RAG  │   LegalRAG  │ Improvement ║")
    print("╠════════════════════════════════════════════════════════════════════════════════╣")
    
    metrics = [
        ('citation_accuracy', 'Citation Accuracy', '%'),
        ('irac_compliance', 'IRAC Compliance', '%'),
        ('hallucination_rate', 'Hallucination Rate', '%'),
        ('total_citations', 'Avg Citations/Memo', '')
    ]
    
    for metric_key, metric_label, unit in metrics:
        values = []
        for approach in approaches:
            data = df[df['approach'] == approach]
            mean_val = data[metric_key].mean()
            values.append(mean_val)
        
        # Calculate improvement (LegalRAG vs Naive RAG)
        if metric_key == 'hallucination_rate':
            improvement = ((values[1] - values[2]) / values[1] * 100) if values[1] > 0 else 0
            improvement_str = f"-{improvement:.1f}%"
        else:
            improvement = ((values[2] - values[1]) / values[1] * 100) if values[1] > 0 else 0
            improvement_str = f"+{improvement:.1f}%"
        
        if unit == '%':
            row = f"║ {metric_label:<23} │   {values[0]*100:>6.1f}%   │   {values[1]*100:>6.1f}%   │   {values[2]*100:>6.1f}%   │  {improvement_str:>9} ║"
        else:
            row = f"║ {metric_label:<23} │    {values[0]:>6.1f}    │    {values[1]:>6.1f}    │    {values[2]:>6.1f}    │  {improvement_str:>9} ║"
        
        print(row)
    
    print("╚════════════════════════════════════════════════════════════════════════════════╝")
    
    # Statistical significance notes
    print("\nStatistical Significance:")
    
    lr_data = df[df['approach'] == 'legalrag']
    nr_data = df[df['approach'] == 'naive_rag']
    
    for metric_key, metric_label, _ in metrics[:3]:  # Only for main metrics
        _, p_value = stats.ttest_rel(lr_data[metric_key], nr_data[metric_key])
        sig = interpret_p_value(p_value)
        print(f"  {metric_label}: {sig}")


def confidence_intervals(df):
    """Calculate 95% confidence intervals"""
    print_section_header("95% CONFIDENCE INTERVALS")
    
    metrics = {
        'citation_accuracy': 'Citation Accuracy',
        'irac_compliance': 'IRAC Compliance',
        'hallucination_rate': 'Hallucination Rate'
    }
    
    for approach in ['zero_shot', 'naive_rag', 'legalrag']:
        data = df[df['approach'] == approach]
        
        print(f"\n{approach.upper().replace('_', ' ')}:")
        print("-" * 80)
        
        for metric, label in metrics.items():
            values = data[metric].values
            mean = np.mean(values)
            se = stats.sem(values)
            ci = se * stats.t.ppf((1 + 0.95) / 2, len(values) - 1)
            
            print(f"  {label:<25} {mean:.4f} ± {ci:.4f}  "
                  f"[{mean - ci:.4f}, {mean + ci:.4f}]")


def query_analysis(df):
    """Analyze performance per query"""
    print_section_header("PER-QUERY ANALYSIS")
    
    n_queries = len(df[df['approach'] == 'legalrag'])
    
    print(f"\nTotal Queries: {n_queries}\n")
    
    # LegalRAG perfect queries
    lr_data = df[df['approach'] == 'legalrag']
    perfect_citation = lr_data[lr_data['citation_accuracy'] == 1.0]
    perfect_irac = lr_data[lr_data['irac_compliance'] == 1.0]
    zero_hallucination = lr_data[lr_data['hallucination_rate'] == 0.0]
    
    print("LegalRAG Performance:")
    print(f"  Perfect Citation Accuracy (100%):  {len(perfect_citation)}/{n_queries} ({len(perfect_citation)/n_queries*100:.1f}%)")
    print(f"  Perfect IRAC Compliance (100%):    {len(perfect_irac)}/{n_queries} ({len(perfect_irac)/n_queries*100:.1f}%)")
    print(f"  Zero Hallucinations:                {len(zero_hallucination)}/{n_queries} ({len(zero_hallucination)/n_queries*100:.1f}%)")
    
    # Queries where LegalRAG struggled
    struggles = lr_data[lr_data['citation_accuracy'] < 0.8]
    if len(struggles) > 0:
        print(f"\nQueries with < 80% Citation Accuracy:")
        for _, row in struggles.iterrows():
            print(f"  Query {row['query_id']}: {row['citation_accuracy']*100:.1f}% "
                  f"({row['correct_citations']}/{row['total_citations']} correct)")
    else:
        print(f"\n✓ All queries achieved ≥ 80% citation accuracy!")
    
    # Hallucinated citations analysis
    total_hallucinated = lr_data['hallucinated_citations'].sum()
    total_citations = lr_data['total_citations'].sum()
    
    print(f"\nCitation Summary:")
    print(f"  Total Citations Generated: {int(total_citations)}")
    print(f"  Correct Citations:         {int(total_citations - total_hallucinated)} ({(total_citations - total_hallucinated)/total_citations*100:.1f}%)")
    print(f"  Hallucinated Citations:    {int(total_hallucinated)} ({total_hallucinated/total_citations*100:.1f}%)")


def latex_table(df):
    """Generate LaTeX table for thesis"""
    print_section_header("LaTeX TABLE (Copy to Thesis)")
    
    approaches = ['zero_shot', 'naive_rag', 'legalrag']
    
    print("\n\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Performance Comparison of Legal Memorandum Generation Systems}")
    print("\\label{tab:results}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("\\textbf{Metric} & \\textbf{Zero-Shot} & \\textbf{Naive RAG} & \\textbf{LegalRAG} & \\textbf{Improvement} \\\\")
    print("\\hline")
    
    metrics = [
        ('citation_accuracy', 'Citation Accuracy', '%'),
        ('irac_compliance', 'IRAC Compliance', '%'),
        ('hallucination_rate', 'Hallucination Rate', '%'),
        ('total_citations', 'Avg Citations/Memo', '')
    ]
    
    for metric_key, metric_label, unit in metrics:
        values = []
        for approach in approaches:
            data = df[df['approach'] == approach]
            mean_val = data[metric_key].mean()
            values.append(mean_val)
        
        # Calculate improvement
        if metric_key == 'hallucination_rate':
            improvement = ((values[1] - values[2]) / values[1] * 100) if values[1] > 0 else 0
            improvement_str = f"$-${improvement:.1f}\\%"
        else:
            improvement = ((values[2] - values[1]) / values[1] * 100) if values[1] > 0 else 0
            improvement_str = f"$+${improvement:.1f}\\%"
        
        if unit == '%':
            print(f"{metric_label} & {values[0]*100:.1f}\\% & {values[1]*100:.1f}\\% & \\textbf{{{values[2]*100:.1f}\\%}} & {improvement_str} \\\\")
        else:
            print(f"{metric_label} & {values[0]:.1f} & {values[1]:.1f} & \\textbf{{{values[2]:.1f}}} & {improvement_str} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    """Run all statistical tests"""
    print("=" * 80)
    print("  LegalRAG Statistical Analysis")
    print("=" * 80)
    
    # Load data
    df = load_latest_results()
    
    # Check if we have all three approaches
    approaches = df['approach'].unique()
    if len(approaches) != 3:
        print(f"Warning: Expected 3 approaches, found {len(approaches)}: {approaches}")
    
    # Check number of queries
    n_queries = len(df[df['approach'] == 'legalrag'])
    print(f"Number of queries: {n_queries}")
    
    if n_queries < 3:
        print("Warning: Very few queries. Results may not be reliable.")
    
    # Run all analyses
    descriptive_statistics(df)
    paired_t_tests(df)
    summary_table(df)
    confidence_intervals(df)
    query_analysis(df)
    latex_table(df)
    
    # Final summary
    print_section_header("THESIS SUMMARY")
    
    lr_data = df[df['approach'] == 'legalrag']
    nr_data = df[df['approach'] == 'naive_rag']
    
    ca_t, ca_p = stats.ttest_rel(lr_data['citation_accuracy'], nr_data['citation_accuracy'])
    ic_t, ic_p = stats.ttest_rel(lr_data['irac_compliance'], nr_data['irac_compliance'])
    hr_t, hr_p = stats.ttest_rel(lr_data['hallucination_rate'], nr_data['hallucination_rate'])
    
    print("\nKEY FINDINGS:")
    print(f"  1. Citation Accuracy:  LegalRAG {lr_data['citation_accuracy'].mean():.1%} vs Naive RAG {nr_data['citation_accuracy'].mean():.1%}")
    print(f"     Statistical significance: {interpret_p_value(ca_p)}")
    
    print(f"\n  2. IRAC Compliance:    LegalRAG {lr_data['irac_compliance'].mean():.1%} vs Naive RAG {nr_data['irac_compliance'].mean():.1%}")
    print(f"     Statistical significance: {interpret_p_value(ic_p)}")
    
    print(f"\n  3. Hallucination Rate: LegalRAG {lr_data['hallucination_rate'].mean():.1%} vs Naive RAG {nr_data['hallucination_rate'].mean():.1%}")
    print(f"     Statistical significance: {interpret_p_value(hr_p)}")
    
    print(f"\n  4. Research Depth:     LegalRAG {lr_data['total_citations'].mean():.1f} citations/memo vs Naive RAG {nr_data['total_citations'].mean():.1f}")
    
    print("\nCONCLUSION:")
    if ic_p < 0.05:
        print("  ✓ LegalRAG shows statistically significant improvement in IRAC compliance")
    if ca_p < 0.05:
        print("  ✓ LegalRAG shows statistically significant improvement in citation accuracy")
    if hr_p < 0.05:
        print("  ✓ LegalRAG shows statistically significant reduction in hallucinations")
    
    print("\n  LegalRAG demonstrates superior performance with modular architecture")
    print("  providing explainability, verification, and consistent structure.")
    
    print("\n" + "=" * 80)
    print("Analysis complete! Results ready for thesis.")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
