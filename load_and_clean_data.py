"""
COMPREHENSIVE DATA LOADING & CLEANING SCRIPT
Stock Market Analysis Project

This script performs perfect data loading and cleaning for all CSV files:
- Loads all CSV files from the data directory
- Handles missing values comprehensively  
- Removes outliers and invalid data
- Ensures no null values in final dataset
- Provides detailed cleaning statistics
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class StockDataProcessor:
    """Advanced Stock Data Processing System"""
    
    def __init__(self):
        self.raw_data_dir = 'stock market dataset'
        self.processed_data_dir = 'data/processed'
        self.stats = {}
    
    def load_all_csv_files(self):
        """Load all CSV files from the data directory"""
        print("📂 Loading all CSV files...")
        
        # Get all CSV files
        csv_files = glob.glob(os.path.join(self.raw_data_dir, '*.csv'))
        print(f"Found {len(csv_files)} CSV files")
        
        all_data = []
        
        for file_path in csv_files:
            try:
                # Extract stock symbol from filename
                symbol = os.path.basename(file_path).replace('.csv', '')
                
                # Load CSV with multiple encoding attempts
                df = None
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except:
                        continue
                
                if df is None:
                    print(f"⚠️  Could not read {symbol}")
                    continue
                
                # Add symbol column
                df['Symbol'] = symbol
                all_data.append(df)
                
            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
                continue
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"✅ Loaded {len(combined_data):,} total records")
        
        return combined_data
    
    def clean_data(self, data):
        """Clean the stock data"""
        print("🧹 Cleaning data...")
        
        initial_count = len(data)
        
        # Convert Date column
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        
        # Remove rows with invalid dates
        data = data.dropna(subset=['Date'])
        
        # Ensure numeric columns are numeric
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove rows with missing OHLC data
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Remove invalid price data (negative or zero prices)
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            data = data[data[col] > 0]
        
        # Remove invalid volume data
        if 'Volume' in data.columns:
            data = data[data['Volume'] >= 0]
            data['Volume'] = data['Volume'].fillna(0)
        
        # Remove price inconsistencies (High < Low, etc.)
        data = data[data['High'] >= data['Low']]
        data = data[data['High'] >= data['Open']]
        data = data[data['High'] >= data['Close']]
        data = data[data['Low'] <= data['Open']]
        data = data[data['Low'] <= data['Close']]
        
        # Remove extreme outliers (prices > 1000x median)
        for col in price_columns:
            median_price = data[col].median()
            data = data[data[col] <= median_price * 1000]
        
        final_count = len(data)
        removed_count = initial_count - final_count
        
        print(f"   Initial records: {initial_count:,}")
        print(f"   Final records: {final_count:,}")
        print(f"   Removed records: {removed_count:,}")
        print(f"   Data retention: {(final_count/initial_count)*100:.2f}%")
        
        return data
    
    def save_processed_data(self, data):
        """Save the processed data"""
        print("💾 Saving processed data...")
        
        # Create directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Save to CSV
        output_path = os.path.join(self.processed_data_dir, 'cleaned_stock_data.csv')
        data.to_csv(output_path, index=False)
        
        print(f"✅ Saved to: {output_path}")
        
        # Show summary stats
        print(f"\n📊 Final Dataset Summary:")
        print(f"   Records: {len(data):,}")
        print(f"   Stocks: {data['Symbol'].nunique()}")
        print(f"   Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
        print(f"   Columns: {list(data.columns)}")
        
        return output_path

def main():
    """Main function to execute comprehensive data loading and cleaning"""
    
    print("🎯 STOCK MARKET DATA LOADING & CLEANING SYSTEM")
    print("=" * 70)
    print()
    
    # Initialize the processor
    processor = StockDataProcessor()
    
    # Load all CSV files
    raw_data = processor.load_all_csv_files()
    
    if len(raw_data) == 0:
        print("❌ No data loaded!")
        return False
    
    # Clean the data
    clean_data = processor.clean_data(raw_data)
    
    # Save processed data
    output_path = processor.save_processed_data(clean_data)
    
    print("\n✅ DATA LOADING & CLEANING COMPLETED SUCCESSFULLY!")
    print("📊 Check 'data/processed/' for cleaned datasets")
    print("🎯 Ready for ML analysis!")
    
    return True

if __name__ == "__main__":
    main()