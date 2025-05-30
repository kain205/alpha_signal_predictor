import pandas as pd
import numpy as np
import os
import glob
import time
from vnstock import Vnstock

# Global Settings 
OUTPUT_FOLDER = "calculated_stock_features"
REQUEST_DELAY_SECONDS = 1.25 
stock_instance_main = Vnstock()

def map_quarter_to_date(row):
    try:
        year = int(row['yearReport'])
        quarter = int(row['lengthReport'])

        if quarter == 1:
            return pd.Timestamp(f"{year}-03-31")
        elif quarter == 2:
            return pd.Timestamp(f"{year}-06-30")
        elif quarter == 3:
            return pd.Timestamp(f"{year}-09-30")
        elif quarter == 4:
            return pd.Timestamp(f"{year}-12-31")
        else:
            return pd.NaT  # invalid quarter number
    except (ValueError, TypeError, KeyError):
        return pd.NaT

def calculate_features(df_stock_raw, ticker_name_for_api):
    df = df_stock_raw.copy()

    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    # Updated fundamental feature names
    fundamental_feature_names_flat = ['FEAT_DebtEquity_quarterly', 'FEAT_PE_quarterly', 'FEAT_EVEBITDA_quarterly', 'FEAT_ROE_quarterly']
    for col_name in fundamental_feature_names_flat: df[col_name] = np.nan

    if stock_instance_main:
        try:
            print(f"    Fetching quarterly ratios for {ticker_name_for_api}...")
            time.sleep(REQUEST_DELAY_SECONDS)
            stock_specific_instance = stock_instance_main.stock(symbol=ticker_name_for_api, source='VCI')
            df_quarterly_ratios_raw = stock_specific_instance.finance.ratio(
                period='quarter', lang='en', dropna=False
            )
            df_flat_columns = pd.DataFrame()

            if not df_quarterly_ratios_raw.empty:
                if isinstance(df_quarterly_ratios_raw.columns, pd.MultiIndex):
                    print(f"    Quarterly data for {ticker_name_for_api} has MultiIndex columns. Flattening...")
                    temp_flat_data = {}
                    for col_tuple in df_quarterly_ratios_raw.columns:
                        metric_name = col_tuple[-1]
                        if metric_name not in temp_flat_data:
                            temp_flat_data[metric_name] = df_quarterly_ratios_raw[col_tuple].values
                    df_flat_columns = pd.DataFrame(temp_flat_data, index=df_quarterly_ratios_raw.index)
                else:
                    print(f"    Quarterly data for {ticker_name_for_api} has simple Index columns.")
                    df_flat_columns = df_quarterly_ratios_raw.copy()

                universal_cols_map = {
                    'Debt/Equity': 'FEAT_DebtEquity_quarterly',
                    'P/E': 'FEAT_PE_quarterly',
                    'EV/EBITDA': 'FEAT_EVEBITDA_quarterly',
                    'ROE (%)': 'FEAT_ROE_quarterly',
                    'yearReport': 'yearReport',
                    'lengthReport': 'lengthReport'
                }
                
                df_fundamentals_selected = pd.DataFrame(index=df_flat_columns.index)
                all_required_flat_cols_found = True

                for source_col_name, target_col_name in universal_cols_map.items():
                    if source_col_name in df_flat_columns.columns:
                        df_fundamentals_selected[target_col_name] = df_flat_columns[source_col_name]
                    else:
                        print(f"    [Warning] Universal column '{source_col_name}' not found in processed quarterly data for {ticker_name_for_api}.")
                        df_fundamentals_selected[target_col_name] = np.nan
                        if source_col_name in ['yearReport', 'lengthReport']:
                            all_required_flat_cols_found = False

                if all_required_flat_cols_found and not df_fundamentals_selected.empty:
                    for col in universal_cols_map.keys():
                        if col in df_fundamentals_selected.columns:
                            if not pd.api.types.is_numeric_dtype(df_fundamentals_selected[col]):
                                df_fundamentals_selected[col] = pd.to_numeric(df_fundamentals_selected[col], errors='coerce')

                    df_fundamentals_selected = df_fundamentals_selected.dropna(subset=['yearReport', 'lengthReport'])
                
                    if not df_fundamentals_selected.empty:
                        df_fundamentals_selected['time'] = df_fundamentals_selected.apply(map_quarter_to_date, axis=1)
                        df_fundamentals_selected = df_fundamentals_selected.dropna(subset=['time'])
                        df_fundamentals_selected = df_fundamentals_selected.sort_values(by='time')

                        cols_to_merge_final = ['time'] + [col_name for col_name in fundamental_feature_names_flat if col_name in df_fundamentals_selected.columns]

                        if not df_fundamentals_selected.empty and 'time' in df_fundamentals_selected.columns and len(cols_to_merge_final) > 1:
                            df_fundamentals_to_merge = df_fundamentals_selected[cols_to_merge_final].drop_duplicates(subset=['time'], keep='last')
                            df_fundamentals_to_merge = df_fundamentals_to_merge.set_index('time')
                            right_ffilled = df_fundamentals_to_merge.reindex(df.index).ffill()
                            df.update(right_ffilled)
                            df = df.reset_index()
                            print(f"    Successfully merged quarterly fundamental data for {ticker_name_for_api}.")
                        else:
                            print(f"    Fundamental data for {ticker_name_for_api} empty after date processing/column selection for merge. Using NaNs.")
                    else:
                        print(f"    Fundamental data for {ticker_name_for_api} missing critical yearReport/lengthReport after coercion. Using NaNs.")
                else:
                    print(f"    Essential 'yearReport' or 'lengthReport' missing or df_fundamentals_selected empty for {ticker_name_for_api}. Using NaNs for fundamental features.")
            else:
                print(f"    Quarterly ratio data for {ticker_name_for_api} is empty from source. Using NaNs for fundamental features.")
        except Exception as e:
            print(f"    [Error] Failed to fetch or process quarterly ratios for {ticker_name_for_api}: {e}")

    else:
        print(f"    vnstock not available, skipping fundamental data for {ticker_name_for_api}.")

    df['daily_return'] = df['close'].pct_change()
    for n in [5, 10, 20]: df[f'price_mom_{n}d'] = df['close'].pct_change(periods=n) * 100
    df['vol_weighted_daily_ret'] = df['daily_return'] * df['volume']
    for n in [5, 10, 20]: df[f'vol_w_mom_{n}d'] = df['vol_weighted_daily_ret'].rolling(window=n, min_periods=n).sum()
    df['volatility_10d'] = df['daily_return'].rolling(window=10, min_periods=10).std()
    df['skewness_20d'] = df['daily_return'].rolling(window=20, min_periods=20).skew()
    delta = df['close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(com=13, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))
    df['rsi_14d'] = df['rsi_14d'].replace([np.inf, -np.inf], np.nan)
    df.loc[(avg_loss == 0) & (avg_gain > 0), 'rsi_14d'] = 100
    df.loc[(avg_loss == 0) & (avg_gain == 0), 'rsi_14d'] = 50
    df['price_mom_10d_raw'] = df['close'].pct_change(periods=10)
    rolling_mean_mom = df['price_mom_10d_raw'].rolling(window=60, min_periods=60).mean()
    rolling_std_mom = df['price_mom_10d_raw'].rolling(window=60, min_periods=60).std()
    df['zscore_mom_10d_60w'] = (df['price_mom_10d_raw'] - rolling_mean_mom) / rolling_std_mom
    df['zscore_mom_10d_60w'] = df['zscore_mom_10d_60w'].replace([np.inf, -np.inf], np.nan)
    df['FUT_RET_10D'] = (df['close'].shift(-10) / df['close'] - 1) * 100

    technical_feature_cols = [col for col in df.columns if 'price_mom_' in col and col != 'price_mom_10d_raw'] + \
                             [col for col in df.columns if 'vol_w_mom_' in col] + \
                             ['volatility_10d', 'skewness_20d', 'rsi_14d', 'zscore_mom_10d_60w']
    final_fundamental_cols = [col for col in fundamental_feature_names_flat if col in df.columns]
    output_columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'daily_return'] + \
                     final_fundamental_cols + technical_feature_cols + ['FUT_RET_10D']
    existing_output_columns = [col for col in output_columns if col in df.columns]
    df_out = df[existing_output_columns].copy()
    key_technical_and_target_cols_for_dropna = ['zscore_mom_10d_60w', 'FUT_RET_10D'] + fundamental_feature_names_flat
    df_out = df_out.dropna(subset=[col for col in key_technical_and_target_cols_for_dropna if col in df_out.columns])
    return df_out

def main():
    current_script_directory = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    input_base_folder = os.path.join(current_script_directory, "stock_data")
    input_folder = os.path.join(input_base_folder, "individual_stocks")
    output_folder_path = os.path.join(current_script_directory, OUTPUT_FOLDER)
    os.makedirs(output_folder_path, exist_ok= True);
    exchanges = ["HOSE", "HNX"]
    for ex in exchanges:
        path = os.path.join(output_folder_path, ex)
        os.makedirs(path, exist_ok=True) 
    print(f"Created output folder: {output_folder_path}")

    stock_files = glob.glob(os.path.join(input_folder, "*", "*.csv"))    
    if not stock_files: print(f"No CSV files found in {input_folder}"); return
    print(f"Found {len(stock_files)} stock files to process.")

    for stock_file_path in stock_files:
        ticker_name_from_file = os.path.basename(stock_file_path).replace(".csv", "")
        exchange = os.path.basename(os.path.dirname(stock_file_path))

        #prepare input features
        try:
            df_stock_raw = pd.read_csv(stock_file_path)
            df_features = calculate_features(df_stock_raw, ticker_name_for_api=ticker_name_from_file)
            if not df_features.empty:
                output_file_path_full = os.path.join(output_folder_path, exchange, f"{ticker_name_from_file}_features.csv")
                df_features.to_csv(output_file_path_full, index=False)
                print(f"    Successfully processed and saved features to: {output_file_path_full}")
            else:
                print(f"    No features generated for {ticker_name_from_file} (data issues or insufficient history).")
        except Exception as e:
            print(f"    [Critical Error] Failed to process file {stock_file_path}: {e}")
        


    print("\n--- Feature calculation process complete! ---")
    print(f"Calculated features saved in: {output_folder_path}")

if __name__ == "__main__":
    main()