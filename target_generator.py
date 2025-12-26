#!/usr/bin/env python3
"""
Target Variable Generation Module
==================================

Creates target variables for:
1. Regression: Future stock prices
2. Classification: BUY/SELL/HOLD labels
3. Multiple prediction horizons

Ensures NO DATA LEAKAGE by using only future data for targets.

Author: Stock Prediction System
Date: December 2025
"""

import pandas as pd
import numpy as np


class TargetGenerator:
    """Generate target variables for stock prediction."""
    
    def __init__(self, data):
        """
        Initialize with prepared data.
        
        Args:
            data (pd.DataFrame): Prepared stock data with features
        """
        self.data = data.copy()
        
    def create_regression_target(self, horizon=5):
        """
        Create regression target: Future stock price.
        
        Args:
            horizon (int): Days ahead to predict (default 5)
            
        Returns:
            pd.DataFrame: Data with future price target
        """
        print(f"\n🎯 CREATING REGRESSION TARGET (Horizon={horizon} days)")
        print("-" * 60)
        
        # Create future price for each stock separately
        self.data[f'Target_Price_{horizon}d'] = self.data.groupby('Symbol')['Close'].shift(-horizon)
        
        # Count non-null targets
        valid_targets = self.data[f'Target_Price_{horizon}d'].notna().sum()
        print(f"✅ Created regression target: Target_Price_{horizon}d")
        print(f"   Valid targets: {valid_targets:,}/{len(self.data):,}")
        
        return self.data
    
    def create_classification_target(self, horizon=5, buy_threshold=0.02, sell_threshold=-0.02):
        """
        Create classification target: BUY/SELL/HOLD labels.
        
        Logic:
        - BUY: Future return > +2% (price will increase)
        - SELL: Future return < -2% (price will decrease)
        - HOLD: Future return between -2% and +2% (sideways movement)
        
        Args:
            horizon (int): Days ahead to predict
            buy_threshold (float): % gain threshold for BUY (default 0.02 = 2%)
            sell_threshold (float): % loss threshold for SELL (default -0.02 = -2%)
            
        Returns:
            pd.DataFrame: Data with BUY/SELL/HOLD labels
        """
        print(f"\n🎯 CREATING CLASSIFICATION TARGET (Horizon={horizon} days)")
        print("-" * 60)
        print(f"   BUY threshold: >{buy_threshold*100:.1f}%")
        print(f"   SELL threshold: <{sell_threshold*100:.1f}%")
        print(f"   HOLD: between thresholds")
        
        # Get future price (already created by regression target)
        if f'Target_Price_{horizon}d' not in self.data.columns:
            self.create_regression_target(horizon=horizon)
        
        # Calculate future return
        self.data[f'Future_Return_{horizon}d'] = (
            (self.data[f'Target_Price_{horizon}d'] - self.data['Close']) / self.data['Close']
        )
        
        # Create labels based on thresholds
        def assign_label(future_return):
            if pd.isna(future_return):
                return None
            elif future_return >= buy_threshold:
                return 'BUY'
            elif future_return <= sell_threshold:
                return 'SELL'
            else:
                return 'HOLD'
        
        self.data[f'Target_Action_{horizon}d'] = self.data[f'Future_Return_{horizon}d'].apply(assign_label)
        
        # Show label distribution
        label_counts = self.data[f'Target_Action_{horizon}d'].value_counts()
        total = label_counts.sum()
        
        print(f"\n📊 Label Distribution:")
        for label in ['BUY', 'HOLD', 'SELL']:
            count = label_counts.get(label, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"   {label:5}: {count:,} ({pct:.1f}%)")
        
        # Check for class imbalance
        max_pct = (label_counts.max() / total * 100) if total > 0 else 0
        min_pct = (label_counts.min() / total * 100) if total > 0 else 0
        
        if max_pct / min_pct > 3:
            print(f"\n⚠️  WARNING: Class imbalance detected (ratio {max_pct/min_pct:.1f}:1)")
            print(f"   Will use class weights in classification models")
        else:
            print(f"\n✅ Classes relatively balanced")
        
        return self.data
    
    def create_all_targets(self, horizons=[5, 10, 20]):
        """
        Create targets for multiple prediction horizons.
        
        Args:
            horizons (list): List of horizons in days
        """
        print("\n" + "=" * 60)
        print("CREATING TARGETS FOR MULTIPLE HORIZONS")
        print("=" * 60)
        
        for horizon in horizons:
            self.create_regression_target(horizon=horizon)
            self.create_classification_target(horizon=horizon)
        
        # Drop rows with missing targets
        initial_rows = len(self.data)
        self.data = self.data.dropna(subset=[f'Target_Action_{horizons[0]}d'])
        final_rows = len(self.data)
        
        print(f"\n📊 FINAL DATASET:")
        print(f"   Rows before: {initial_rows:,}")
        print(f"   Rows after: {final_rows:,}")
        print(f"   Removed: {initial_rows - final_rows:,} (no future data available)")
        
        return self.data
    
    def get_feature_target_split(self, horizon=5):
        """
        Split data into features (X) and targets (y).
        
        Args:
            horizon (int): Prediction horizon
            
        Returns:
            tuple: (X, y_regression, y_classification)
        """
        # Exclude columns that shouldn't be features
        exclude_cols = [
            'Date', 'Symbol', 'Series', 'Prev Close', 'Last', 'VWAP', 
            'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble'
        ]
        
        # Also exclude all target columns
        target_cols = [col for col in self.data.columns if 'Target_' in col or 'Future_' in col]
        exclude_cols.extend(target_cols)
        
        # Get feature columns
        feature_cols = [col for col in self.data.columns 
                       if col not in exclude_cols 
                       and self.data[col].dtype in ['int64', 'float64']]
        
        X = self.data[feature_cols]
        y_regression = self.data[f'Target_Price_{horizon}d']
        y_classification = self.data[f'Target_Action_{horizon}d']
        
        print(f"\n📋 FEATURE-TARGET SPLIT:")
        print(f"   Features (X): {X.shape[1]} columns")
        print(f"   Regression target: Target_Price_{horizon}d")
        print(f"   Classification target: Target_Action_{horizon}d")
        
        return X, y_regression, y_classification


def main():
    """Test target generation with sample data."""
    print("🧪 TESTING TARGET GENERATION")
    print("=" * 80)
    
    # Load prepared data
    data = pd.read_csv('data/processed/full_dataset.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    
    print(f"Loaded {len(data):,} records from {data['Symbol'].nunique()} stocks")
    
    # Create targets
    generator = TargetGenerator(data)
    data_with_targets = generator.create_all_targets(horizons=[5])
    
    # Get feature-target split
    X, y_reg, y_class = generator.get_feature_target_split(horizon=5)
    
    print(f"\n✅ TARGET GENERATION COMPLETE")
    print(f"   Ready for model training!")
    
    # Save data with targets
    output_path = 'data/processed/data_with_targets.csv'
    data_with_targets.to_csv(output_path, index=False)
    print(f"\n💾 Saved: {output_path}")


if __name__ == "__main__":
    main()
