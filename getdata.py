import yfinance as yf
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns

def load_stock_data():
    # Lấy dữ liệu stock từ Yahoo Finance
    stock_data = yf.download(tickers="AAPL", period="1y")

    # Xử lý giá trị thiếu (NaN)
    stock_data.dropna(inplace=True)  # Loại bỏ các hàng chứa giá trị NaN

    # Kiểm tra xem có giá trị NaN không
    dem_NaN = stock_data.isnull().sum().sum()
    print("Số lượng dữ liệu NaN:", dem_NaN)

    # Kiểm tra giá trị trùng lặp
    duplicates = stock_data.duplicated()
    if duplicates.any():
        print(f"Có {duplicates.sum()} hàng bị trùng lặp")
    else:
        print("Không có hàng bị trùng lặp")

    # Xử lý dữ liệu ngoại lai
    Q1 = stock_data.quantile(0.25)
    Q3 = stock_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (stock_data < (Q1 - 1.5 * IQR)) | (stock_data > (Q3 + 1.5 * IQR))
    stock_data = stock_data[~outliers.any(axis=1)]

    # Xóa lại các giá trị NaN sau khi loại bỏ ngoại lai
    stock_data.dropna(inplace=True)

    # Thống kê mô tả
    print("Descriptive Statistics:")
    print(stock_data.describe())  # Cung cấp giá trị trung bình, tiêu chuẩn, nhỏ nhất, lớn nhất, tứ phân vị, etc.

    # Đồ thị để trực quan hóa dữ liệu
    stock_data.hist(bins=10, figsize=(10, 6))
    plt.xlabel("Feature")
    plt.ylabel("Frequency")
    plt.title("Distribution of Stock Features")
    plt.subplots_adjust(bottom=0.1)
    plt.show()

    # Chuẩn bị dữ liệu cho mô hình
    features = stock_data[['Open', 'High', 'Low', 'Volume']]
    target = stock_data['Close']

    print("Features shape:", features.shape)
    print("Target shape:", target.shape)

    return features, target

# Gọi hàm để kiểm tra
print(load_stock_data())










