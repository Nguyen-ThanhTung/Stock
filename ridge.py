
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from getdata import load_stock_data
# Tải dữ liệu cổ phiếu

features, target = load_stock_data()

# Chia dữ liệu thành 70% train và 30% còn lại
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)

# Chia tập còn lại thành 15% test và 15% validation
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

# Scale data
scaler = MinMaxScaler()
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
print("MSE:",mse_train)
print("MAE:",mae_train)
print("R2:",r2_train)

print("Chỉ số đánh giá trên tập val")
mse_val = mean_squared_error(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)
print("MSE:",mse_val)
print("MAE:",mae_val)
print("R2:",r2_val)

print("Chỉ số đánh giá trên tập test")
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
print("MSE:",mse_test)
print("MAE:",mae_test)
print("R2:",r2_test)

# Vẽ biểu đồ cho tập train
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, alpha=0.5, label='Train Data', color='blue')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Train Set: Actual vs Predicted")
plt.grid(True)
plt.legend()
plt.show()

# Vẽ biểu đồ cho tập validation
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_val_pred, alpha=0.5, label='Validation Data', color='green')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Validation Set: Actual vs Predicted")
plt.grid(True)
plt.legend()
plt.show()

# Vẽ biểu đồ cho tập test
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5, label='Test Data', color='orange')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Test Set: Actual vs Predicted")
plt.grid(True)
plt.legend()
plt.show()