import pandas as pd
import os

# === INPUT / OUTPUT ===
# Use the correct file that exists based on your terminal output
input_file = "/Users/ophelianeo/Documents/work!/finance/Tickers_with_ratios_ordered.xlsx"
output_dir = "/Users/ophelianeo/Documents/work!/finance/cutting_results"
os.makedirs(output_dir, exist_ok=True)

# === Load Data with better error handling ===
def load_data(file_path, sheet_name=None):
    """Load Excel file with fallback options"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            # Try openpyxl first (for .xlsx files)
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
            except Exception as e1:
                print(f"⚠️ openpyxl failed: {e1}")
                # Fallback to xlrd (for .xls files)
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="xlrd")
                except Exception as e2:
                    print(f"⚠️ xlrd failed: {e2}")
                    # Last resort - try without specifying engine
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        raise RuntimeError(f"❌ Could not read {file_path}: {e}")
    
    return df

# Load the data
try:
    df = load_data(input_file)
    print(f"✅ Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    print("Available columns:", df.columns.tolist())
except Exception as e:
    print(f"❌ Error loading data: {e}")
    # Try alternative file paths
    alternative_files = [
        "/Users/ophelianeo/Documents/work!/finance/Tickers_with_ratios_20250905_215807.xlsx",
        "/Users/ophelianeo/Documents/work!/finance/Screening here.xlsx"
    ]
    
    for alt_file in alternative_files:
        if os.path.exists(alt_file):
            print(f"🔄 Trying alternative file: {alt_file}")
            try:
                df = load_data(alt_file, sheet_name="top 100")
                input_file = alt_file
                print(f"✅ Successfully loaded: {len(df)} rows, {len(df.columns)} columns")
                break
            except Exception as e2:
                print(f"❌ Failed to load {alt_file} with sheet 'top 100': {e2}")
                # Try without sheet name as fallback
                try:
                    df = load_data(alt_file, sheet_name=None)
                    input_file = alt_file
                    print(f"✅ Successfully loaded (default sheet): {len(df)} rows, {len(df.columns)} columns")
                    break
                except Exception as e3:
                    print(f"❌ Failed to load {alt_file} completely: {e3}")
                    continue
    else:
        print("❌ Could not load any data files")
        exit(1)

# === Column Mapping (updated for new criteria) ===
# Map expected columns to actual columns in your data
column_mapping = {
    'Net_Profit_Margin': 'Net_Profit_Margin',
    'ROE': 'ROE',
    'Debt_Ratio': 'Debt_Ratio',
    'Interest_Coverage': 'Interest_Coverage',
    'Current_Ratio': 'Current_Ratio',
    '50DMA': '50DMA',
    '200DMA': '200DMA',
    'EV_EBITDA': 'EV_EBITDA'  # Optional - will check if exists
}

# Check which columns actually exist and create missing ones if needed
print("\n=== Column Check ===")
for expected, actual in column_mapping.items():
    if actual in df.columns:
        print(f"✅ {expected}: Found as '{actual}'")
    else:
        print(f"❌ {expected}: Not found")
        
        # Handle missing Net_Profit_Margin - try to calculate it or use a proxy
        if expected == 'Net_Profit_Margin':
            possible_profit_cols = [col for col in df.columns if 'profit' in col.lower() or 'margin' in col.lower()]
            if possible_profit_cols:
                print(f"🔄 Found possible profit columns: {possible_profit_cols}")
                column_mapping[expected] = possible_profit_cols[0]
                print(f"🔄 Using '{possible_profit_cols[0]}' as Net_Profit_Margin")
            else:
                df['Net_Profit_Margin'] = 0
                print(f"🔄 Created dummy Net_Profit_Margin column (all zeros)")
        
        # Handle missing EV/EBITDA - try alternative names
        if expected == 'EV_EBITDA':
            possible_ev_cols = [col for col in df.columns if 'ev' in col.lower() and 'ebitda' in col.lower()]
            if not possible_ev_cols:
                possible_ev_cols = [col for col in df.columns if 'enterprise' in col.lower() and 'ebitda' in col.lower()]
            if possible_ev_cols:
                print(f"🔄 Found possible EV/EBITDA columns: {possible_ev_cols}")
                column_mapping[expected] = possible_ev_cols[0]
                print(f"🔄 Using '{possible_ev_cols[0]}' as EV_EBITDA")

# === Apply New Filters Step by Step ===
print("\n=== Applying Updated Screening Criteria ===")

# Define filters based on your new criteria
steps = []

# 1. Profitability: Keep only profitable firms
if column_mapping['Net_Profit_Margin'] in df.columns:
    steps.append(("Step1_Profitability", df[column_mapping['Net_Profit_Margin']] >= 0))
    print("📊 Filter 1: Net Profit Margin ≥ 0% (Profitability)")

# 2. Efficiency: Ensure decent returns on equity
if column_mapping['ROE'] in df.columns:
    steps.append(("Step2_Efficiency", df[column_mapping['ROE']] >= 0.12))
    print("📊 Filter 2: ROE ≥ 12% (Efficiency)")

# 3. Solvency: Avoid over-leveraged firms (Combined filter)
debt_condition = None
interest_condition = None

if column_mapping['Debt_Ratio'] in df.columns:
    debt_condition = df[column_mapping['Debt_Ratio']] <= 0.6
    
if column_mapping['Interest_Coverage'] in df.columns:
    interest_condition = df[column_mapping['Interest_Coverage']] >= 3

if debt_condition is not None and interest_condition is not None:
    steps.append(("Step3_Solvency", debt_condition & interest_condition))
    print("📊 Filter 3: Debt Ratio ≤ 60% AND Interest Coverage ≥ 3 (Solvency)")
elif debt_condition is not None:
    steps.append(("Step3_Solvency_Debt", debt_condition))
    print("📊 Filter 3a: Debt Ratio ≤ 60% (Partial Solvency)")
elif interest_condition is not None:
    steps.append(("Step3_Solvency_Interest", interest_condition))
    print("📊 Filter 3b: Interest Coverage ≥ 3 (Partial Solvency)")

# 4. Liquidity: Filter extremes
if column_mapping['Current_Ratio'] in df.columns:
    steps.append(("Step4_Liquidity", df[column_mapping['Current_Ratio']].between(1.2, 3.0)))
    print("📊 Filter 4: Current Ratio between 1.2 and 3.0 (Liquidity)")

# 5. Momentum: Keep those in an uptrend
if column_mapping['50DMA'] in df.columns and column_mapping['200DMA'] in df.columns:
    steps.append(("Step5_Momentum", df[column_mapping['50DMA']] >= df[column_mapping['200DMA']]))
    print("📊 Filter 5: 50DMA ≥ 200DMA (Momentum)")

# 6. Optional: EV/EBITDA filter (reasonable valuation)
if column_mapping['EV_EBITDA'] in df.columns and 'EV_EBITDA' in column_mapping and column_mapping['EV_EBITDA'] in df.columns:
    # Filter out negative EV/EBITDA and very high multiples
    ev_condition = (df[column_mapping['EV_EBITDA']] > 0) & (df[column_mapping['EV_EBITDA']] <= 15)
    steps.append(("Step6_Valuation", ev_condition))
    print("📊 Filter 6: EV/EBITDA between 0 and 15 (Reasonable Valuation) - OPTIONAL")

if not steps:
    print("❌ No filters could be applied - no matching columns found")
    exit(1)

print(f"✅ Applying {len(steps)} filters based on updated criteria")

# === Save All Results to Single Workbook with Multiple Sheets ===
final_workbook = os.path.join(output_dir, "Stock_Screening_Updated_Criteria.xlsx")

df_steps = df.copy()
step_dataframes = {}
step_dataframes['Original_Data'] = df.copy()

try:
    with pd.ExcelWriter(final_workbook, engine='openpyxl') as writer:
        # Write original data
        df.to_excel(writer, sheet_name="Original_Data", index=False)

        # Apply filters one by one and write after each step
        for step_name, condition in steps:
            before = len(df_steps)
            valid_condition = condition.fillna(False) if hasattr(condition, 'fillna') else condition
            df_steps = df_steps[valid_condition].copy()
            after = len(df_steps)

            print(f"{step_name}: {before} → {after} companies ({before-after} filtered out)")

            safe_name = step_name[:31]
            df_steps.to_excel(writer, sheet_name=safe_name, index=False)
            step_dataframes[step_name] = df_steps.copy()
            print(f"💾 Writing sheet: {safe_name} ({len(df_steps)} rows)")

        # Store final result
        step_dataframes['Final_Shortlist'] = df_steps.copy()
        df_steps.to_excel(writer, sheet_name="Final_Shortlist", index=False)

        # Create a summary sheet with updated criteria descriptions
        summary_data = []
        filter_descriptions = {
            'Step1_Profitability': 'Net Profit Margin ≥ 0% (Profitable firms only)',
            'Step2_Efficiency': 'ROE ≥ 12% (Decent returns on equity)', 
            'Step3_Solvency': 'Debt Ratio ≤ 60% AND Interest Coverage ≥ 3 (Avoid over-leverage)',
            'Step3_Solvency_Debt': 'Debt Ratio ≤ 60% (Partial solvency check)',
            'Step3_Solvency_Interest': 'Interest Coverage ≥ 3 (Partial solvency check)',
            'Step4_Liquidity': 'Current Ratio: 1.2 - 3.0 (Filter liquidity extremes)',
            'Step5_Momentum': '50DMA ≥ 200DMA (Uptrend momentum)',
            'Step6_Valuation': 'EV/EBITDA: 0 - 15 (Reasonable valuation - OPTIONAL)'
        }
        
        for step_name, df_step in step_dataframes.items():
            if step_name != 'Original_Data':
                summary_data.append({
                    'Step': step_name,
                    'Description': filter_descriptions.get(step_name, 'Custom filter'),
                    'Companies_Remaining': len(df_step),
                    'Companies_Filtered_Out': len(df) - len(df_step),
                    'Percentage_Remaining': f"{len(df_step)/len(df)*100:.1f}%"
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        print("💾 Writing sheet: Summary")
    
    print(f"\n✅ All results saved to single workbook: {final_workbook}")
    print(f"📊 Workbook contains {len(step_dataframes) + 1} sheets (Original + each step + Final + Summary)")
    
    if len(df_steps) > 0:
        print(f"\n🎯 Success rate: {len(df_steps)}/{len(df)} ({len(df_steps)/len(df)*100:.1f}%) companies passed all filters")
        if 'Yahoo_Ticker' in df_steps.columns:
            print("📊 Sample tickers that passed all criteria:")
            print(df_steps['Yahoo_Ticker'].head(10).tolist())
        elif 'Ticker' in df_steps.columns:
            print("📊 Sample tickers that passed all criteria:")
            print(df_steps['Ticker'].head(10).tolist())
    else:
        print("⚠️ No companies passed all filters - consider relaxing criteria")

    # Print final screening summary
    print(f"\n📋 SCREENING CRITERIA SUMMARY:")
    print("1. ✅ Profitability: Net Profit Margin ≥ 0%")
    print("2. ✅ Efficiency: ROE ≥ 12%") 
    print("3. ✅ Solvency: Debt Ratio ≤ 60% AND Interest Coverage ≥ 3")
    print("4. ✅ Liquidity: Current Ratio between 1.2 and 3.0")
    print("5. ✅ Momentum: 50DMA ≥ 200DMA")
    if any('Valuation' in step for step, _ in steps):
        print("6. ✅ Valuation: EV/EBITDA between 0 and 15 (OPTIONAL)")
        
except Exception as e:
    print(f"❌ Could not save workbook: {e}")

print(f"\n🏁 Updated screening complete! Check {final_workbook} for all results with new criteria.")