import pandas as pd
import os
import numpy as np
from datetime import datetime

# === INPUT / OUTPUT ===
input_file = "/Users/ophelianeo/Documents/work!/finance/Screening here.xlsx"
output_dir = "/Users/ophelianeo/Documents/work!/finance/cutting_results"
os.makedirs(output_dir, exist_ok=True)

# === Configuration ===
SCREENING_CRITERIA = {
    'profitability': {'Net_Profit_Margin': 0.0},
    'roe': {'ROE': 0.12},
    'solvency': {'Debt_Ratio': 0.6, 'Interest_Coverage': 3.0},
    'liquidity': {'Current_Ratio_min': 1.2, 'Current_Ratio_max': 3.0},
    'momentum': {}  # 50DMA >= 200DMA
}

# === Enhanced Load Data Function ===
def load_data(file_path, sheet_name=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            return pd.read_csv(file_path)
        else:
            try:
                if sheet_name:
                    print(f"📄 Loading sheet: '{sheet_name}'")
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
                return df
            except Exception as e1:
                print(f"⚠️  openpyxl failed: {e1}")
                try:
                    excel_file = pd.ExcelFile(file_path, engine="openpyxl")
                    sheet_names = excel_file.sheet_names
                    print(f"📋 Available sheets: {sheet_names}")
                    if sheet_name and sheet_name not in sheet_names:
                        sheet_name = sheet_names[0]
                        print(f"📄 Using sheet: '{sheet_name}' instead")
                    elif not sheet_name:
                        sheet_name = sheet_names[0]
                        print(f"📄 Using first sheet: '{sheet_name}'")
                    return pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
                except Exception as e2:
                    print(f"⚠️  Fallback to xlrd: {e2}")
                    try:
                        return pd.read_excel(file_path, sheet_name=sheet_name, engine="xlrd")
                    except Exception as e3:
                        print(f"⚠️  xlrd failed, trying default: {e3}")
                        return pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        raise RuntimeError(f"❌ Could not read {file_path}: {e}")

def try_alternative_files():
    alternative_files = [
        "/Users/ophelianeo/Documents/work!/finance/Tickers_with_ratios_ordered.xlsx",
        "/Users/ophelianeo/Documents/work!/finance/Tickers_with_ratios_20250905_215807.xlsx"
    ]
    for alt_file in alternative_files:
        if os.path.exists(alt_file):
            print(f"🔄 Trying alternative file: {alt_file}")
            try:
                df = load_data(alt_file)
                return df, alt_file
            except Exception as e:
                print(f"❌ Failed to load {alt_file}: {e}")
                continue
    return None, None

def check_and_handle_columns(df):
    print("\n=== Column Analysis ===")
    column_mapping = {
        'Net_Profit_Margin': ['Net_Profit_Margin', 'Profit_Margin', 'Net_Margin'],
        '50DMA': ['50DMA', '50_day_MA', 'MA_50'],
        '200DMA': ['200DMA', '200_day_MA', 'MA_200'],
        'ROE': ['ROE', 'Return_on_Equity'],
        'ROA': ['ROA', 'Return_on_Assets'],
        'Debt_Ratio': ['Debt_Ratio', 'Total_Debt_Ratio'],
        'Interest_Coverage': ['Interest_Coverage', 'Interest_Coverage_Ratio'],
        'Quick_Ratio': ['Quick_Ratio', 'Acid_Test_Ratio'],
        'Current_Ratio': ['Current_Ratio'],
        'Debt_to_Equity': ['Debt_to_Equity', 'D_E_Ratio']
    }
    final_mapping = {}
    missing_columns = []
    for expected, alternatives in column_mapping.items():
        found = False
        for alt in alternatives:
            if alt in df.columns:
                final_mapping[expected] = alt
                print(f"✅ {expected}: Found as '{alt}'")
                found = True
                break
        if not found:
            possible_matches = [col for col in df.columns 
                                if any(keyword.lower() in col.lower() 
                                       for keyword in expected.split('_'))]
            if possible_matches:
                final_mapping[expected] = possible_matches[0]
                print(f"🔄 {expected}: Using similar column '{possible_matches[0]}'")
            else:
                missing_columns.append(expected)
                print(f"❌ {expected}: Not found")
    for missing in missing_columns:
        if missing == 'Net_Profit_Margin':
            df['Net_Profit_Margin'] = 0.01
            final_mapping[missing] = 'Net_Profit_Margin'
            print(f"🔄 Created dummy {missing} column (assuming profitability)")
    return final_mapping, df

def apply_screening_step(df, step_name, condition, description, current_step, total_steps):
    initial_count = len(df)
    if hasattr(condition, 'fillna'):
        valid_condition = condition.fillna(False)
    else:
        valid_condition = condition
    filtered_df = df[valid_condition].copy()
    final_count = len(filtered_df)
    eliminated = initial_count - final_count
    elimination_rate = (eliminated / initial_count * 100) if initial_count > 0 else 0
    print(f"\n[{current_step}/{total_steps}] {step_name}")
    print(f"  📊 {initial_count} → {final_count} companies ({eliminated} eliminated)")
    print(f"  📉 Elimination rate: {elimination_rate:.1f}%")
    print(f"  🎯 Criteria: {description}")
    return filtered_df

# === Main Execution ===
try:
    print("🔍 Starting Enhanced Stock Screening Process...")
    try:
        df = load_data(input_file, sheet_name="Sheet4")
        loaded_file = input_file
        print(f"✅ Loaded primary file: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"⚠️  Primary file failed: {e}")
        print("🔄 Trying alternative files...")
        df, loaded_file = try_alternative_files()
        if df is None:
            print("❌ Could not load any data files")
            exit(1)
        else:
            print(f"✅ Loaded alternative file: {len(df)} rows, {len(df.columns)} columns")
    column_mapping, df = check_and_handle_columns(df)
    if not column_mapping:
        print("❌ No usable columns found")
        exit(1)
    print(f"✅ Column mapping complete: {len(column_mapping)} columns mapped")
except Exception as e:
    print(f"❌ Error in data preparation: {e}")
    exit(1)

print("\n" + "="*60)
print("🎯 FUNDAMENTAL ANALYSIS SCREENING PROCESS")
print("="*60)

# Build screening steps
screening_steps = []
if 'Net_Profit_Margin' in column_mapping:
    screening_steps.append((
        '1_Profitability_Filter',
        df[column_mapping['Net_Profit_Margin']] >= SCREENING_CRITERIA['profitability']['Net_Profit_Margin'],
        f"Net Profit Margin ≥ {SCREENING_CRITERIA['profitability']['Net_Profit_Margin']:.0%}"
    ))
if 'ROE' in column_mapping:
    screening_steps.append((
        '2_ROE_Quality',
        df[column_mapping['ROE']] >= SCREENING_CRITERIA['roe']['ROE'],
        f"ROE ≥ {SCREENING_CRITERIA['roe']['ROE']:.0%}"
    ))
if 'Debt_Ratio' in column_mapping and 'Interest_Coverage' in column_mapping:
    screening_steps.append((
        '3_Financial_Strength',
        (df[column_mapping['Debt_Ratio']] <= SCREENING_CRITERIA['solvency']['Debt_Ratio']) &
        (df[column_mapping['Interest_Coverage']] >= SCREENING_CRITERIA['solvency']['Interest_Coverage']),
        f"Debt Ratio ≤ {SCREENING_CRITERIA['solvency']['Debt_Ratio']:.0%} AND Interest Coverage ≥ {SCREENING_CRITERIA['solvency']['Interest_Coverage']:.0f}x"
    ))
if 'Current_Ratio' in column_mapping:
    screening_steps.append((
        '4_Liquidity_Health',
        df[column_mapping['Current_Ratio']].between(
            SCREENING_CRITERIA['liquidity']['Current_Ratio_min'],
            SCREENING_CRITERIA['liquidity']['Current_Ratio_max']
        ),
        f"Current Ratio {SCREENING_CRITERIA['liquidity']['Current_Ratio_min']}-{SCREENING_CRITERIA['liquidity']['Current_Ratio_max']}"
    ))
if '50DMA' in column_mapping and '200DMA' in column_mapping:
    screening_steps.append((
        '5_Technical_Momentum',
        df[column_mapping['50DMA']] >= df[column_mapping['200DMA']],
        "50DMA ≥ 200DMA"
    ))
if not screening_steps:
    print("❌ No screening steps could be created - insufficient column data")
    exit(1)

print(f"📋 Configured {len(screening_steps)} screening steps")

# === Save results step by step ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_workbook = os.path.join(output_dir, f"Enhanced_Stock_Screening_{timestamp}.xlsx")

step_dataframes = {}
df_current = df.copy()
with pd.ExcelWriter(final_workbook, engine='openpyxl') as writer:
    # Export original data
    step_dataframes['0_Original_Data'] = df.copy()
    df.to_excel(writer, sheet_name='0_Original_Data', index=False)

    total_steps = len(screening_steps)
    for i, (step_name, condition, description) in enumerate(screening_steps, 1):
        df_current = apply_screening_step(df_current, step_name, condition, description, i, total_steps)
        step_dataframes[step_name] = df_current.copy()
        safe_sheet_name = step_name[:31]
        df_current.to_excel(writer, sheet_name=safe_sheet_name, index=False)
        print(f"💾 Exported: {safe_sheet_name} ({len(df_current)} companies)")

    # Final candidates
    step_dataframes['6_Final_Investment_Candidates'] = df_current.copy()
    df_current.to_excel(writer, sheet_name='6_Final_Investment_Candidates', index=False)

    # Screening summary
    summary_data = []
    original_count = len(step_dataframes['0_Original_Data'])
    for step_name, df_step in step_dataframes.items():
        if step_name.startswith('0_'):
            continue
        remaining = len(df_step)
        eliminated = original_count - remaining
        summary_data.append({
            'Screening_Step': step_name.replace('_', ' '),
            'Companies_Remaining': remaining,
            'Companies_Eliminated': eliminated,
            'Survival_Rate': f"{remaining/original_count*100:.1f}%",
            'Cumulative_Elimination': f"{eliminated/original_count*100:.1f}%"
        })
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Screening_Summary', index=False)

    # Top performers
    if len(df_current) > 0:
        ranking_columns = []
        if 'ROE' in column_mapping and column_mapping['ROE'] in df_current.columns:
            ranking_columns.append(column_mapping['ROE'])
        if 'Net_Profit_Margin' in column_mapping and column_mapping['Net_Profit_Margin'] in df_current.columns:
            ranking_columns.append(column_mapping['Net_Profit_Margin'])
        if ranking_columns:
            top_performers = df_current.nlargest(min(25, len(df_current)), ranking_columns[0])
            top_performers.to_excel(writer, sheet_name='Top_Performers', index=False)
            print(f"💾 Exported: Top_Performers ({len(top_performers)} companies)")

# === Final Summary ===
print("\n" + "="*60)
print("🏆 SCREENING PROCESS COMPLETE!")
print("="*60)
final_count = len(df_current)
original_count = len(df)
survival_rate = final_count / original_count * 100
print(f"📁 Results file: {final_workbook}")
print(f"📊 Original companies: {original_count:,}")
print(f"🎯 Final candidates: {final_count:,}")
print(f"📈 Success rate: {survival_rate:.2f}%")
