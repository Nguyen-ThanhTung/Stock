from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
import yfinance as yf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

app = Flask(__name__)

# Global scaler for use across models and predictions
scaler = MinMaxScaler()


# Function to load stock data and clean NaNs and outliers
def load_stock_data():
    # Lấy dữ liệu stock từ Yahoo
    stock_data = yf.download(tickers="AAPL", period="1y")

    # Handle missing values consistently
    stock_data.dropna(inplace=True)  # Drop rows with any missing values

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

    # Chuẩn bị dữ liệu cho mô hình
    features = stock_data[['Open', 'High', 'Low', 'Volume']]
    target = stock_data['Close']

    print("Features shape:", features.shape)
    print("Target shape:", target.shape)

    return features, target


def train_linear_model():
    features, target = load_stock_data()
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Dự đoán cho các tập dữ liệu
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Các chỉ số đánh giá
    print("Chỉ số đánh giá trên tập train:")
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    print(mse_train)
    print(mae_train)
    print(r2_train)

    print("Chỉ số đánh giá trên tập val")
    mse_val = mean_squared_error(y_val, y_val_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    print(mse_val)
    print(mae_val)
    print(r2_val)

    print("Chỉ số đánh giá trên tập test")
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print(mse_test)
    print(mae_test)
    print(r2_test)
    return model, mae_test, mse_test, r2_test


def train_ridge_model():
    features, target = load_stock_data()
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

    # Scale data
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    model = RidgeCV(alphas=alphas)
    model.fit(X_train_scaled, y_train)

    # Dự đoán cho các tập dữ liệu
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Các chỉ số đánh giá
    print("Chỉ số đánh giá trên tập train:")
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    print(mse_train)
    print(mae_train)
    print(r2_train)

    print("Chỉ số đánh giá trên tập val")
    mse_val = mean_squared_error(y_val, y_val_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    print(mse_val)
    print(mae_val)
    print(r2_val)

    print("Chỉ số đánh giá trên tập test")
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print(mse_test)
    print(mae_test)
    print(r2_test)

    return model, mae_test, mse_test, r2_test


# Function to train Neural Network model
def train_neural_model():
    features, target = load_stock_data()
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    mlp = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=4000, random_state=42)
    mlp.fit(X_train_scaled, y_train)

    # Dự đoán cho các tập dữ liệu
    y_train_pred = mlp.predict(X_train_scaled)
    y_val_pred = mlp.predict(X_val_scaled)
    y_test_pred = mlp.predict(X_test_scaled)

    # Các chỉ số đánh giá
    print("Chỉ số đánh giá trên tập train:")
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    print(mse_train)
    print(mae_train)
    print(r2_train)

    print("Chỉ số đánh giá trên tập val")
    mse_val = mean_squared_error(y_val, y_val_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    print(mse_val)
    print(mae_val)
    print(r2_val)

    print("Chỉ số đánh giá trên tập test")
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print(mse_test)
    print(mae_test)
    print(r2_test)

    return mlp, mae_test, mse_test, r2_test


# Function to train Stacking model
def train_stacking_model():
    features, target = load_stock_data()

    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Define base models
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=4000, random_state=42)
    ridge = RidgeCV(alphas=[0.0001, 0.001 , 0.01, 0.1, 1, 10, 100])
    linreg = LinearRegression()

    # Define stacking model
    stacking_model = StackingRegressor(
        estimators=[('mlp', mlp), ('ridge', ridge), ('linreg', linreg)],
        final_estimator=RidgeCV()
    )

    # Train stacking model
    stacking_model.fit(X_train_scaled, y_train)

    # Evaluate on validation data
    # Dự đoán cho các tập dữ liệu
    y_train_pred = stacking_model.predict(X_train_scaled)
    y_val_pred = stacking_model.predict(X_val_scaled)
    y_test_pred = stacking_model.predict(X_test_scaled)

    # Các chỉ số đánh giá
    print("Chỉ số đánh giá trên tập train:")
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    print("MSE:", mse_train)
    print("MAE:", mae_train)
    print("R2:", r2_train)

    print("Chỉ số đánh giá trên tập val")
    mse_val = mean_squared_error(y_val, y_val_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    print("MSE:", mse_val)
    print("MAE:", mae_val)
    print("R2:", r2_val)

    print("Chỉ số đánh giá trên tập test")
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print("MSE:", mse_test)
    print("MAE:", mae_test)
    print("R2:", r2_test)

    return stacking_model, mse_test, mae_test, r2_test


# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')


# Route to handle prediction and return evaluation metrics
@app.route('/predict', methods=['POST'])
def predict():
    try:
        open_price = float(request.form['open'])
        high_price = float(request.form['high'])
        low_price = float(request.form['low'])
        volume = float(request.form['volume'])

        selected_model = request.form['model_type']

        if selected_model == 'linear':
            model, mse, mae, r2 = train_linear_model()
        elif selected_model == 'ridge':
            model, mse, mae, r2 = train_ridge_model()
        elif selected_model == 'neural':
            model, mse, mae, r2 = train_neural_model()
        elif selected_model == 'stacking':
            model, mse, mae, r2 = train_stacking_model()
        else:
            return render_template('index.html', prediction_text="Invalid model selection")

        # Prepare input data for prediction and scale if necessary
        input_features = np.array([[open_price, high_price, low_price, volume]])
        if selected_model != 'linear':  # Apply scaling for Ridge, Neural, and Stacking models
            input_features = scaler.transform(input_features)

        predicted_close_price = model.predict(input_features)[0]

        return render_template('index.html',
                               prediction_text=f'Predicted Closing Price: ${predicted_close_price}',
                               mse=f'MSE: {mse}',
                               mae=f'MAE: {mae}',
                               r2=f'R²: {r2}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run()

