import os
import pandas as pd
from vnstock import Vnstock
from datetime import datetime, timedelta
import time

# --- Cài đặt chung ---
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime('%Y-%m-%d')
REQUEST_DELAY_SECONDS = 1

stock_instance = Vnstock()

# --- Hàm tiện ích ---
def fetch_and_save_stock_data(ticker, exchange, start_date, end_date, folder_path, data_source='VCI'):
    folder_path = os.path.join(folder_path, exchange)
    try:
        print(f"Đang chuẩn bị tải dữ liệu cho mã: {ticker}...")
        time.sleep(REQUEST_DELAY_SECONDS)
        print(f"--> Đang tải dữ liệu cho mã: {ticker} từ {start_date} đến {end_date}...")

        df = stock_instance.stock(symbol=ticker, source=data_source).quote.history(
            start=start_date,
            end=end_date,
            interval='1D'
        )
        if df is not None and not df.empty:
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values(by='time')
            
            file_path = os.path.join(folder_path, f"{ticker}.csv")
            df.to_csv(file_path, index=False)
            print(f"    Đã lưu dữ liệu cho {ticker} vào {file_path}")
        else:
            print(f"    Không có dữ liệu cho mã {ticker} trong khoảng thời gian đã chọn hoặc DataFrame trả về rỗng.")
    except Exception as e:
        print(f"    Lỗi khi tải dữ liệu cho mã {ticker}: {e}")

# --- Tạo thư mục lưu trữ ---
OUTPUT_FOLDER = "stock_data"
market_data_folder = os.path.join(OUTPUT_FOLDER, "market_indices")
dirs_to_create = [
    OUTPUT_FOLDER,
    os.path.join(OUTPUT_FOLDER, "individual_stocks"),
    os.path.join(OUTPUT_FOLDER, "market_indices"),
]
stock_data_folder = os.path.join(OUTPUT_FOLDER, "individual_stocks")
for folder in dirs_to_create:
    os.makedirs(folder, exist_ok=True)
exchanges = ["HOSE", "HNX"]
for ex in exchanges:
    path = os.path.join(OUTPUT_FOLDER, "individual_stocks", ex)
    os.makedirs(path, exist_ok=True)


# --- Lấy danh sách mã cổ phiếu từ HOSE và HNX ---
print("Đang lấy danh sách mã cổ phiếu...")
hose_tickers = []
hnx_tickers = []
all_symbols_df = None

try:
    listing_obj = stock_instance.stock(symbol='ACB', source='VCI').listing
    all_symbols_df_raw = listing_obj.symbols_by_exchange() # Lấy dữ liệu thô

    if all_symbols_df_raw is not None and not all_symbols_df_raw.empty:
        print("--- Thông tin DataFrame thô từ symbols_by_exchange() ---")
        print("Các cột có trong DataFrame thô:")
        print(all_symbols_df_raw.columns)
        print(f"Số lượng mã ban đầu: {len(all_symbols_df_raw)}")
        print(f"Các giá trị duy nhất trong cột 'type' (ban đầu): {all_symbols_df_raw['type'].unique()}")

        if 'type' in all_symbols_df_raw.columns:
            all_symbols_df = all_symbols_df_raw[all_symbols_df_raw['type'] == 'STOCK'].copy()
            print(f"Số lượng mã sau khi lọc theo 'type' == 'STOCK': {len(all_symbols_df)}")
        else:
            print("LỖI: Không tìm thấy cột 'type' để lọc. Sử dụng DataFrame thô.")
            all_symbols_df = all_symbols_df_raw.copy()
        # ***********************************************

        if all_symbols_df is not None and not all_symbols_df.empty: # Kiểm tra lại sau khi lọc
            # print("\n5 dòng đầu của DataFrame sau khi lọc 'STOCK':") # Có thể bỏ comment để xem
            # print(all_symbols_df.head())

            TICKER_COLUMN_NAME = 'symbol'
            EXCHANGE_COLUMN_NAME = 'exchange'

            if TICKER_COLUMN_NAME in all_symbols_df.columns and EXCHANGE_COLUMN_NAME in all_symbols_df.columns:
                print(f"\nCác giá trị duy nhất trong cột '{EXCHANGE_COLUMN_NAME}' (sau khi lọc STOCK): {all_symbols_df[EXCHANGE_COLUMN_NAME].unique()}")

                hose_tickers = all_symbols_df[all_symbols_df[EXCHANGE_COLUMN_NAME] == 'HSX'][TICKER_COLUMN_NAME].unique().tolist()
                hnx_tickers = all_symbols_df[all_symbols_df[EXCHANGE_COLUMN_NAME] == 'HNX'][TICKER_COLUMN_NAME].unique().tolist()

                print(f"\nTổng số mã trên HOSE (đã lọc STOCK): {len(hose_tickers)}")
                print(f"5 mã HOSE đầu tiên: {hose_tickers[:5] if hose_tickers else 'Không có'}")
                print(f"Tổng số mã trên HNX (đã lọc STOCK): {len(hnx_tickers)}")
                print(f"5 mã HNX đầu tiên: {hnx_tickers[:5] if hnx_tickers else 'Không có'}")
            else:
                print(f"LỖI: Cột '{TICKER_COLUMN_NAME}' hoặc '{EXCHANGE_COLUMN_NAME}' không tồn tại trong DataFrame đã lọc.")
        else:
            print("DataFrame rỗng sau khi cố gắng lọc theo 'type' == 'STOCK'.")

    else:
        print("Không lấy được dữ liệu từ symbols_by_exchange() hoặc DataFrame trả về rỗng.")

except Exception as e:
    print(f"Lỗi nghiêm trọng khi lấy danh sách mã cổ phiếu: {e}")
    if all_symbols_df_raw is not None: # In thông tin từ df thô nếu có lỗi
        print("Thông tin all_symbols_df_raw tại thời điểm lỗi:")
        print(all_symbols_df_raw.head())
        print(all_symbols_df_raw.columns)


# --- Tải và lưu dữ liệu cho từng mã cổ phiếu ---
for hose_ticker in hose_tickers:
    fetch_and_save_stock_data(hose_ticker, "HOSE", START_DATE, END_DATE, stock_data_folder, data_source='VCI')
for hnx_ticker in hnx_tickers:
    fetch_and_save_stock_data(hnx_ticker, "HNX", START_DATE, END_DATE, stock_data_folder, data_source='VCI')

# --- Tải và lưu dữ liệu chỉ số thị trường ---
print("\n--- Bắt đầu tải dữ liệu chỉ số thị trường ---")
market_indices = {
    "VNINDEX": "HOSE",
    "HNXINDEX": "HNX"
}

for index_ticker, exchange_name in market_indices.items():
    print(f"Đang chuẩn bị tải dữ liệu cho chỉ số: {index_ticker} ({exchange_name})")
    time.sleep(REQUEST_DELAY_SECONDS)
    print(f"--> Đang tải dữ liệu cho chỉ số: {index_ticker} ({exchange_name})")
    try:
        df_index = stock_instance.stock(symbol=index_ticker, source='VCI').quote.history(
            start=START_DATE,
            end=END_DATE,
            interval='1D'
        )
        if df_index is not None and not df_index.empty:
            if 'time' in df_index.columns:
                df_index['time'] = pd.to_datetime(df_index['time'])
                df_index = df_index.sort_values(by='time')
            file_path = os.path.join(market_data_folder, f"{index_ticker}.csv")
            df_index.to_csv(file_path, index=False)
            print(f"    Đã lưu dữ liệu cho chỉ số {index_ticker} vào {file_path}")
        else:
            print(f"    Không có dữ liệu cho chỉ số {index_ticker} hoặc DataFrame trả về rỗng.")
    except Exception as e:
        print(f"    Lỗi khi tải dữ liệu cho chỉ số {index_ticker}: {e}")

print("\n--- Quá trình thu thập dữ liệu hoàn tất! ---")
print(f"Dữ liệu cổ phiếu riêng lẻ được lưu tại: {stock_data_folder}")
print(f"Dữ liệu chỉ số thị trường được lưu tại: {market_data_folder}")