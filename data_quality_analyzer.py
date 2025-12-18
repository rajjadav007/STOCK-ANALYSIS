#!/usr/bin/env python3
"""
Advanced Data Quality Analysis for Stock Market Data
=====================================================

This module performs comprehensive data quality validation specifically designed 
for financial time series data. It goes beyond basic checks to identify issues
that could impact ML model performance.

Key Areas Analyzed:
1. Missing Values Pattern Analysis
2. Outlier Detection (Financial Context)
3. Date Range Validation & Gap Analysis
4. Price Logic Validation
5. Volume Anomaly Detection
6. Data Distribution Analysis

Author: Stock Market ML Analysis Project
Date: December 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from pathlib import Path

# Import our data loader
from data_loader import StockDataLoader

warnings.filterwarnings('ignore')

class StockDataQualityAnalyzer:
    """
    Advanced Data Quality Analysis System for Financial Data
    
    This class performs enterprise-grade data quality validation that goes
    beyond basic pandas checks to identify financial data specific issues.
    """
    
    def __init__(self):
        self.quality_report = {}
        self.outliers_detected = {}
        self.date_issues = {}
        
    def analyze_missing_values_pattern(self, df, symbol):
        """
        Advanced missing value analysis for financial data
        
        Financial data has specific patterns:
        - Early data often missing newer fields
        - Holiday periods may have gaps
        - Corporate actions can cause data anomalies
        """
        print(f"\n🔍 MISSING VALUES PATTERN ANALYSIS: {symbol}")
        print("=" * 50)
        
        missing_data = df.isnull().sum()
        total_rows = len(df)
        
        # Create missing value report
        missing_report = {}
        
        for col in df.columns:
            missing_count = missing_data[col]
            missing_pct = (missing_count / total_rows) * 100
            
            if missing_count > 0:
                # Analyze WHERE missing values occur
                missing_indices = df[df[col].isnull()].index.tolist()
                
                # Check if missing values are clustered (common in financial data)
                if len(missing_indices) > 1:
                    gaps = np.diff(missing_indices)
                    consecutive_gaps = np.sum(gaps == 1)
                    scattered_gaps = len(missing_indices) - consecutive_gaps
                else:
                    consecutive_gaps = 0
                    scattered_gaps = len(missing_indices)
                
                missing_report[col] = {
                    'count': missing_count,
                    'percentage': missing_pct,
                    'consecutive_gaps': consecutive_gaps,
                    'scattered_gaps': scattered_gaps,
                    'first_missing': df[df[col].isnull()]['Date'].min() if 'Date' in df.columns else 'N/A',
                    'last_missing': df[df[col].isnull()]['Date'].max() if 'Date' in df.columns else 'N/A'
                }
                
                print(f"\n📊 {col}:")
                print(f"   Missing: {missing_count:,} ({missing_pct:.2f}%)")
                print(f"   Pattern: {consecutive_gaps} consecutive, {scattered_gaps} scattered")
                if 'Date' in df.columns:
                    print(f"   Date Range: {missing_report[col]['first_missing']} to {missing_report[col]['last_missing']}")
                
                # Financial data specific advice
                if col in ['Trades', 'Deliverable Volume']:
                    print(f"   💡 Note: {col} data often unavailable in early years - this is normal")
                elif missing_pct > 20:
                    print(f"   ⚠️  WARNING: High missing rate for {col} - consider dropping this column")
        
        if not missing_report:
            print("✅ No missing values found - Excellent data quality!")
        
        return missing_report
    
    def detect_financial_outliers(self, df, symbol):
        """
        Detect outliers specific to financial data
        
        Financial outliers include:
        - Price gaps > 20% (stock splits, corporate actions)
        - Volume spikes > 10x normal
        - Impossible price relationships (High < Low)
        - Zero or negative prices
        """
        print(f"\n🎯 FINANCIAL OUTLIERS DETECTION: {symbol}")
        print("=" * 50)
        
        outliers_found = {}
        
        # 1. Check for impossible price relationships
        print("🔍 Checking Price Logic...")
        impossible_prices = df[
            (df['High'] < df['Low']) | 
            (df['Open'] < 0) | 
            (df['Close'] < 0) | 
            (df['High'] < 0) | 
            (df['Low'] < 0)
        ]
        
        if len(impossible_prices) > 0:
            outliers_found['impossible_prices'] = len(impossible_prices)
            print(f"   ❌ Found {len(impossible_prices)} rows with impossible price relationships!")
            print("   First few problematic rows:")
            print(impossible_prices[['Date', 'Open', 'High', 'Low', 'Close']].head())
        else:
            print("   ✅ All price relationships are logically correct")
        
        # 2. Detect massive price gaps (potential stock splits/bonuses)
        print("\n🔍 Checking for Price Gaps...")
        df['Price_Change_Pct'] = df['Close'].pct_change() * 100
        
        # Price changes > 50% in a day (very rare without corporate actions)
        large_gaps = df[abs(df['Price_Change_Pct']) > 50]
        
        if len(large_gaps) > 0:
            outliers_found['price_gaps'] = len(large_gaps)
            print(f"   ⚠️  Found {len(large_gaps)} days with >50% price changes:")
            for _, row in large_gaps.head().iterrows():
                print(f"      {row['Date'].date()}: {row['Price_Change_Pct']:.1f}% change")
            print("   💡 Note: These might be stock splits/bonuses - verify with corporate actions")
        else:
            print("   ✅ No extreme price gaps detected")
        
        # 3. Volume outliers (spikes > 5x median volume)
        print("\n🔍 Checking Volume Anomalies...")
        median_volume = df['Volume'].median()
        volume_threshold = median_volume * 5
        
        volume_spikes = df[df['Volume'] > volume_threshold]
        
        if len(volume_spikes) > 0:
            outliers_found['volume_spikes'] = len(volume_spikes)
            print(f"   📈 Found {len(volume_spikes)} days with volume >5x median:")
            print(f"      Median Volume: {median_volume:,.0f}")
            print(f"      Threshold: {volume_threshold:,.0f}")
            print("   Top 5 volume spike days:")
            top_volume = volume_spikes.nlargest(5, 'Volume')[['Date', 'Volume', 'Close', 'Price_Change_Pct']]
            for _, row in top_volume.iterrows():
                print(f"      {row['Date'].date()}: {row['Volume']:,.0f} shares ({row['Price_Change_Pct']:.1f}% price change)")
        else:
            print("   ✅ No extreme volume spikes detected")
        
        # 4. Check for zero volume days (market holidays or data errors)
        zero_volume_days = df[df['Volume'] == 0]
        if len(zero_volume_days) > 0:
            outliers_found['zero_volume'] = len(zero_volume_days)
            print(f"\n   ⚠️  Found {len(zero_volume_days)} days with zero volume")
            print("   These might be holidays or data errors")
        
        return outliers_found
    
    def validate_date_ranges(self, df, symbol):
        """
        Comprehensive date range validation for financial data
        
        Checks for:
        - Weekend trading (impossible)
        - Missing trading days
        - Future dates
        - Date sequence integrity
        """
        print(f"\n📅 DATE RANGE VALIDATION: {symbol}")
        print("=" * 50)
        
        date_issues = {}
        
        if 'Date' not in df.columns:
            print("❌ No Date column found!")
            return {'no_date_column': True}
        
        # Basic date range info
        start_date = df['Date'].min()
        end_date = df['Date'].max()
        total_days = (end_date - start_date).days
        trading_days = len(df)
        
        print(f"📊 Date Range Overview:")
        print(f"   Start Date: {start_date.date()}")
        print(f"   End Date: {end_date.date()}")
        print(f"   Calendar Days: {total_days:,}")
        print(f"   Trading Days: {trading_days:,}")
        print(f"   Coverage: {(trading_days/total_days)*100:.1f}%")
        
        # 1. Check for weekend trading (impossible in most markets)
        print(f"\n🔍 Weekend Trading Check:")
        df['Weekday'] = df['Date'].dt.day_name()
        weekend_trading = df[df['Weekday'].isin(['Saturday', 'Sunday'])]
        
        if len(weekend_trading) > 0:
            date_issues['weekend_trading'] = len(weekend_trading)
            print(f"   ❌ Found {len(weekend_trading)} weekend trading days!")
            print("   First few weekend dates:")
            for date in weekend_trading['Date'].head():
                print(f"      {date.strftime('%Y-%m-%d %A')}")
        else:
            print("   ✅ No weekend trading detected")
        
        # 2. Check for future dates
        print(f"\n🔍 Future Dates Check:")
        today = pd.Timestamp.now()
        future_dates = df[df['Date'] > today]
        
        if len(future_dates) > 0:
            date_issues['future_dates'] = len(future_dates)
            print(f"   ❌ Found {len(future_dates)} future dates!")
        else:
            print("   ✅ No future dates detected")
        
        # 3. Check for large gaps in trading days
        print(f"\n🔍 Trading Day Gaps Analysis:")
        df_sorted = df.sort_values('Date')
        df_sorted['Date_Diff'] = df_sorted['Date'].diff().dt.days
        
        # Gaps > 10 days (might indicate missing data or market closures)
        large_gaps = df_sorted[df_sorted['Date_Diff'] > 10]
        
        if len(large_gaps) > 0:
            date_issues['large_gaps'] = len(large_gaps)
            print(f"   ⚠️  Found {len(large_gaps)} gaps >10 days:")
            for _, row in large_gaps.head().iterrows():
                gap_days = int(row['Date_Diff'])
                print(f"      {row['Date'].date()}: {gap_days} day gap")
            print("   💡 Note: These might be market closures or missing data")
        else:
            print("   ✅ No large trading gaps detected")
        
        # 4. Expected vs Actual trading days
        print(f"\n🔍 Trading Days Coverage:")
        # Rough estimate: ~252 trading days per year (excluding weekends/holidays)
        years = (end_date - start_date).days / 365.25
        expected_trading_days = years * 252
        coverage_pct = (trading_days / expected_trading_days) * 100
        
        print(f"   Years of Data: {years:.1f}")
        print(f"   Expected Trading Days: ~{expected_trading_days:.0f}")
        print(f"   Actual Trading Days: {trading_days}")
        print(f"   Coverage: {coverage_pct:.1f}%")
        
        if coverage_pct < 85:
            print(f"   ⚠️  Low coverage - might be missing significant data")
        elif coverage_pct > 105:
            print(f"   ⚠️  High coverage - might have duplicate or extra dates")
        else:
            print(f"   ✅ Good coverage for stock market data")
        
        return date_issues
    
    def generate_data_quality_summary(self, df, symbol):
        """
        Generate comprehensive data quality summary with actionable insights
        """
        print(f"\n📋 DATA QUALITY SUMMARY: {symbol}")
        print("=" * 60)
        
        # Overall data quality score
        total_possible_values = len(df) * len(df.columns)
        missing_values = df.isnull().sum().sum()
        completeness_score = ((total_possible_values - missing_values) / total_possible_values) * 100
        
        # Price data quality (most critical for ML)
        price_cols = ['Open', 'High', 'Low', 'Close']
        price_completeness = 100
        for col in price_cols:
            if col in df.columns:
                col_completeness = (df[col].count() / len(df)) * 100
                price_completeness = min(price_completeness, col_completeness)
        
        print(f"🏆 QUALITY SCORES:")
        print(f"   Overall Completeness: {completeness_score:.2f}%")
        print(f"   Price Data Quality: {price_completeness:.2f}%")
        print(f"   Volume Data Quality: {(df['Volume'].count()/len(df))*100:.2f}%")
        
        # Risk assessment for ML
        print(f"\n🎯 ML READINESS ASSESSMENT:")
        
        if price_completeness >= 99:
            print("   ✅ EXCELLENT - Ready for ML model training")
        elif price_completeness >= 95:
            print("   ✅ GOOD - Suitable for ML with minor preprocessing")
        elif price_completeness >= 90:
            print("   ⚠️  FAIR - Needs data cleaning before ML")
        else:
            print("   ❌ POOR - Significant data issues need resolution")
        
        # Actionable recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        missing_report = df.isnull().sum()
        for col, missing_count in missing_report[missing_report > 0].items():
            missing_pct = (missing_count / len(df)) * 100
            
            if col in ['Trades', 'Deliverable Volume', '%Deliverble']:
                if missing_pct > 50:
                    print(f"   📝 {col}: Consider dropping (high missing rate: {missing_pct:.1f}%)")
                else:
                    print(f"   📝 {col}: Forward fill or interpolate missing values")
            elif col in price_cols:
                if missing_pct > 5:
                    print(f"   ⚠️  {col}: Critical for ML - investigate missing price data")
                else:
                    print(f"   📝 {col}: Forward fill missing values")
            elif col == 'Volume':
                print(f"   📝 {col}: Use median volume for missing values")
        
        return {
            'completeness_score': completeness_score,
            'price_data_quality': price_completeness,
            'ml_ready': price_completeness >= 95,
            'total_rows': len(df),
            'missing_values': missing_values
        }
    
    def run_complete_analysis(self, symbol="RELIANCE"):
        """
        Run the complete data quality analysis pipeline
        """
        print("🔬 COMPREHENSIVE DATA QUALITY ANALYSIS")
        print("=" * 80)
        print("This analysis will identify all data quality issues that could impact ML performance")
        print("=" * 80)
        
        # Load the data
        loader = StockDataLoader()
        df = loader.load_sample_stock(symbol)
        
        if df is None:
            print(f"❌ Failed to load data for {symbol}")
            return None
        
        # Run all analyses
        missing_analysis = self.analyze_missing_values_pattern(df, symbol)
        outlier_analysis = self.detect_financial_outliers(df, symbol)
        date_analysis = self.validate_date_ranges(df, symbol)
        quality_summary = self.generate_data_quality_summary(df, symbol)
        
        # Store results
        self.quality_report[symbol] = {
            'missing_values': missing_analysis,
            'outliers': outlier_analysis,
            'date_issues': date_analysis,
            'summary': quality_summary
        }
        
        print(f"\n🎉 ANALYSIS COMPLETE for {symbol}")
        print(f"📊 Quality Score: {quality_summary['completeness_score']:.2f}%")
        print(f"🤖 ML Ready: {'Yes' if quality_summary['ml_ready'] else 'No'}")
        
        return self.quality_report[symbol]


def main():
    """
    Run comprehensive data quality analysis
    """
    analyzer = StockDataQualityAnalyzer()
    
    # Analyze RELIANCE (our sample stock)
    result = analyzer.run_complete_analysis("RELIANCE")
    
    if result:
        print(f"\n" + "="*80)
        print("🏁 FINAL RECOMMENDATIONS FOR ML PIPELINE")
        print("="*80)
        
        if result['summary']['ml_ready']:
            print("✅ Data is ready for ML model training!")
            print("📝 Next steps:")
            print("   1. Create technical indicators (SMA, RSI, etc.)")
            print("   2. Engineer lag features for time series")
            print("   3. Split data chronologically (train/test)")
            print("   4. Begin model training")
        else:
            print("⚠️  Data needs preprocessing before ML:")
            print("📝 Required steps:")
            print("   1. Handle missing values in critical columns")
            print("   2. Remove or fix outliers")
            print("   3. Validate date integrity")
            print("   4. Re-run this analysis")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()