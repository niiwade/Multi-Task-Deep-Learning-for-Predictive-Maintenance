#!/usr/bin/env python3
"""
Analysis Script for Machine Failure Predictive Maintenance
Performs data analysis, answers key questions, and generates visualizations
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def setup_environment():
    """Setup output directories and matplotlib"""
    os.makedirs("visualizations", exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    print("✓ Environment setup complete")

def load_data():
    """Load and explore dataset"""
    try:
        # Load train dataset
        df = pd.read_csv('../dataset/train/train.csv')
        print(f"✓ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"✓ Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"⚠ Error loading data: {e}")
        return None

def analyze_data(df):
    """Analyze dataset and answer key questions"""
    
    print("\n" + "="*80)
    print("DATASET ANALYSIS - MACHINE FAILURE PREDICTIVE MAINTENANCE")
    print("="*80)
    
    # Q1: Temperature vs Machine Failure
    print("\nQ1: How does temperature affect machine failure rates?")
    print("-" * 60)
    
    failure_rate = (df['Machine failure'].sum() / len(df)) * 100
    print(f"Overall failure rate: {failure_rate:.2f}%")
    
    # Analyze by temperature quartiles
    df['temp_quartile'] = pd.qcut(df['Process temperature [K]'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    temp_analysis = df.groupby('temp_quartile')['Machine failure'].agg(['sum', 'count', 'mean'])
    temp_analysis.columns = ['Failures', 'Total', 'Failure_Rate']
    temp_analysis['Failure_Rate'] = temp_analysis['Failure_Rate'] * 100
    print("\nFailure rates by temperature quartile:")
    print(temp_analysis)
    
    # Q2: Rotational Speed Effects
    print("\n\nQ2: What is the relationship between rotational speed and failures?")
    print("-" * 60)
    
    df['speed_quartile'] = pd.qcut(df['Rotational speed [rpm]'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    speed_analysis = df.groupby('speed_quartile')['Machine failure'].agg(['sum', 'count', 'mean'])
    speed_analysis.columns = ['Failures', 'Total', 'Failure_Rate']
    speed_analysis['Failure_Rate'] = speed_analysis['Failure_Rate'] * 100
    print("\nFailure rates by speed quartile:")
    print(speed_analysis)
    
    # Q3: Torque & Tool Wear Impact
    print("\n\nQ3: How do torque and tool wear correlate with machine failures?")
    print("-" * 60)
    
    failures = df[df['Machine failure'] == 1]
    healthy = df[df['Machine failure'] == 0]
    
    print(f"\nMean Torque (Failed): {failures['Torque [Nm]'].mean():.2f} Nm")
    print(f"Mean Torque (Healthy): {healthy['Torque [Nm]'].mean():.2f} Nm")
    print(f"Mean Tool Wear (Failed): {failures['Tool wear [min]'].mean():.2f} min")
    print(f"Mean Tool Wear (Healthy): {healthy['Tool wear [min]'].mean():.2f} min")
    
    # Calculate correlation
    torque_corr = df[['Torque [Nm]', 'Tool wear [min]', 'Machine failure']].corr()
    print("\nCorrelation with machine failure:")
    print(torque_corr['Machine failure'].sort_values(ascending=False))
    
    # Q4: Failure Types Comparison
    print("\n\nQ4: What are the distribution of different failure types?")
    print("-" * 60)
    
    failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    failure_counts = {}
    for col in failure_cols:
        if col in df.columns:
            count = df[col].sum()
            failure_counts[col] = count
            print(f"{col}: {int(count)} cases")
    
    total_failures = sum(failure_counts.values())
    print(f"\nTotal specific failures: {int(total_failures)}")
    print(f"Multi-failure cases: {int(df[failure_cols].sum(axis=1).gt(1).sum())}")
    
    return df, {
        'temp_analysis': temp_analysis,
        'speed_analysis': speed_analysis,
        'failure_counts': failure_counts
    }

def create_visualizations(df, analysis_results):
    """Create 4 key visualizations"""
    
    # Visualization 1: Temperature vs Failure
    fig, ax = plt.subplots(figsize=(10, 6))
    temp_analysis = analysis_results['temp_analysis']
    bars = ax.bar(range(len(temp_analysis)), temp_analysis['Failure_Rate'], 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8, edgecolor='black')
    ax.set_xlabel('Process Temperature Quartile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Machine Failure Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Q1: How does temperature affect machine failure rates?', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(temp_analysis)))
    ax.set_xticklabels(temp_analysis.index)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/01_temperature_vs_failure.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 01_temperature_vs_failure.png")
    plt.close()
    
    # Visualization 2: Rotational Speed Effects
    fig, ax = plt.subplots(figsize=(10, 6))
    speed_analysis = analysis_results['speed_analysis']
    bars = ax.bar(range(len(speed_analysis)), speed_analysis['Failure_Rate'],
                   color=['#95E1D3', '#F38181', '#AA96DA', '#FCBAD3'], alpha=0.8, edgecolor='black')
    ax.set_xlabel('Rotational Speed Quartile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Machine Failure Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Q2: What is the relationship between rotational speed and failures?', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(speed_analysis)))
    ax.set_xticklabels(speed_analysis.index)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/02_speed_vs_failure.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 02_speed_vs_failure.png")
    plt.close()
    
    # Visualization 3: Torque & Tool Wear Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Torque distribution
    failed_mask = df['Machine failure'] == 1
    axes[0].hist([df[~failed_mask]['Torque [Nm]'], df[failed_mask]['Torque [Nm]']], 
                 label=['Healthy', 'Failed'], bins=30, alpha=0.7, color=['#45B7D1', '#FF6B6B'])
    axes[0].set_xlabel('Torque [Nm]', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Torque Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Tool Wear distribution
    axes[1].hist([df[~failed_mask]['Tool wear [min]'], df[failed_mask]['Tool wear [min]']], 
                 label=['Healthy', 'Failed'], bins=30, alpha=0.7, color=['#45B7D1', '#FF6B6B'])
    axes[1].set_xlabel('Tool Wear [min]', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Tool Wear Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Q3: How do torque and tool wear correlate with machine failures?', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('visualizations/03_torque_toolwear_impact.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 03_torque_toolwear_impact.png")
    plt.close()
    
    # Visualization 4: Failure Types Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    failure_counts = [df[col].sum() for col in failure_cols]
    labels = ['Tool Wear\nFailure', 'Heat Dissipation\nFailure', 'Power\nFailure', 
              'Overstrain\nFailure', 'Random\nFailure']
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(failure_cols)))
    bars = ax.bar(labels, failure_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    ax.set_title('Q4: What are the distribution of different failure types?', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/04_failure_types.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 04_failure_types.png")
    plt.close()

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("MACHINE FAILURE PREDICTIVE MAINTENANCE - DATA ANALYSIS")
    print("="*80 + "\n")
    
    setup_environment()
    
    # Load data
    df = load_data()
    if df is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)
    
    # Analyze data and answer questions
    df, analysis_results = analyze_data(df)
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    create_visualizations(df, analysis_results)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - visualizations/01_temperature_vs_failure.png")
    print("  - visualizations/02_speed_vs_failure.png")
    print("  - visualizations/03_torque_toolwear_impact.png")
    print("  - visualizations/04_failure_types.png")
    print("\n")

if __name__ == "__main__":
    main()
