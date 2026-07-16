"""
Master Script: Create All Figures for Chapter 2
Run this to generate all figures at once!

Requirements:
pip install matplotlib numpy pandas seaborn
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set global style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

# Color palette
COLORS = {
    'zero_shot': '#FF6B6B',
    'naive_rag': '#4ECDC4',
    'legalrag': '#45B7D1',
    'good': '#2ECC71',
    'warning': '#F5A623',
    'bad': '#FF6B6B'
}

def create_figure_2_4_bar_chart():
    """Figure 2.4: Citation Accuracy Bar Chart"""
    print("Creating Figure 2.4: Bar Chart...")
    
    approaches = ['Zero-Shot', 'Naive RAG', 'LegalRAG']
    accuracy = [32.8, 93.9, 98.5]
    std_dev = [33.2, 21.5, 4.2]
    colors = [COLORS['zero_shot'], COLORS['naive_rag'], COLORS['legalrag']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(approaches, accuracy, color=colors, 
                   yerr=std_dev, capsize=10, alpha=0.85,
                   edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Citation Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Approach', fontsize=13, fontweight='bold')
    ax.set_title('Citation Accuracy Across Approaches', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Add value labels
    for bar, acc, std in zip(bars, accuracy, std_dev):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add n annotation
    ax.text(0.98, 0.02, 'n = 25 queries per approach',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig('Figure_2_4_Citation_Accuracy.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure_2_4_Citation_Accuracy.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 2.4 saved")


def create_figure_2_4b_hallucination_breakdown():
    """Figure 2.4b: Hallucination Types Pie Chart"""
    print("Creating Figure 2.4b: Hallucination Types...")
    
    labels = ['Fabricated Names\n(65%)', 'Real Cases,\nWrong Citation\n(25%)', 
              'Misstated Holdings\n(10%)']
    sizes = [65, 25, 10]
    colors = ['#FF6B6B', '#F5A623', '#FFA07A']
    explode = (0.1, 0.05, 0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                        colors=colors, autopct='%1.0f%%',
                                        startangle=90, textprops={'fontsize': 11})
    
    # Bold percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax.set_title('Types of Hallucinations in Zero-Shot Generation',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('Figure_2_4b_Hallucination_Types.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure_2_4b_Hallucination_Types.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 2.4b saved")


def create_figure_2_4c_evolution_timeline():
    """Figure 2.4c: Evolution of Accuracy Over Time"""
    print("Creating Figure 2.4c: Evolution Timeline...")
    
    years = [2020, 2022, 2024, 2026]
    accuracy = [0, 32.8, 93.9, 98.5]
    labels = ['Pre-LLM', 'Zero-Shot', 'Naive RAG', 'LegalRAG']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(years, accuracy, marker='o', linewidth=3, 
            markersize=12, color=COLORS['legalrag'],
            markeredgecolor='black', markeredgewidth=2)
    
    # Add labels
    for year, acc, label in zip(years, accuracy, labels):
        if acc > 0:
            ax.annotate(f'{acc:.1f}%\n{label}', 
                       xy=(year, acc), 
                       xytext=(0, 15), 
                       textcoords='offset points',
                       ha='center', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='white', 
                                edgecolor='gray', alpha=0.8))
    
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Citation Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Evolution of Citation Accuracy (2020-2026)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(-5, 105)
    ax.set_xlim(2019, 2027)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('Figure_2_4c_Evolution.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure_2_4c_Evolution.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 2.4c saved")


def create_figure_comparison_heatmap():
    """Bonus: Feature Comparison Heatmap"""
    print("Creating Bonus Figure: Feature Heatmap...")
    
    # Data for heatmap
    features = ['Query\nExpansion', 'Hybrid\nRetrieval', 'Cross-encoder\nReranking', 
                'Structured\nGeneration', 'Citation\nVerification']
    systems = ['RAG\n(2020)', 'REALM\n(2020)', 'Query2Doc\n(2022)', 
               'RARR\n(2023)', 'LegalRAG\n(2026)']
    
    # 1 = has feature, 0 = doesn't have
    data = np.array([
        [0, 0, 0, 0, 0],  # RAG
        [0, 0, 0, 0, 0],  # REALM
        [1, 0, 0, 0, 0],  # Query2Doc
        [0, 0, 0, 0, 1],  # RARR
        [1, 1, 1, 1, 1],  # LegalRAG
    ])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(systems)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(systems)
    ax.set_yticklabels(features)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add text annotations
    for i in range(len(systems)):
        for j in range(len(features)):
            text = '✓' if data[i, j] == 1 else '✗'
            color = 'white' if data[i, j] == 1 else 'black'
            ax.text(i, j, text, ha="center", va="center", 
                   color=color, fontsize=16, fontweight='bold')
    
    ax.set_title('Feature Comparison: RAG Systems', 
                 fontsize=15, fontweight='bold', pad=15)
    
    # Highlight LegalRAG column
    rect = Rectangle((-0.5, -0.5), 1, len(features), 
                     linewidth=3, edgecolor='purple', 
                     facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig('Figure_Bonus_Feature_Heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure_Bonus_Feature_Heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Bonus Feature Heatmap saved")


def main():
    """Create all figures"""
    print("\n" + "="*50)
    print("  CREATING ALL FIGURES FOR CHAPTER 2")
    print("="*50 + "\n")
    
    create_figure_2_4_bar_chart()
    create_figure_2_4b_hallucination_breakdown()
    create_figure_2_4c_evolution_timeline()
    create_figure_comparison_heatmap()
    
    print("\n" + "="*50)
    print("  ✓ ALL FIGURES CREATED SUCCESSFULLY!")
    print("="*50)
    print("\nFiles created:")
    print("  • Figure_2_4_Citation_Accuracy.png (and .pdf)")
    print("  • Figure_2_4b_Hallucination_Types.png (and .pdf)")
    print("  • Figure_2_4c_Evolution.png (and .pdf)")
    print("  • Figure_Bonus_Feature_Heatmap.png (and .pdf)")
    print("\nAll figures saved at 300 DPI (print quality)")
    print("Both PNG and PDF versions created")
    print("\nReady to insert into your thesis!")


if __name__ == "__main__":
    main()
