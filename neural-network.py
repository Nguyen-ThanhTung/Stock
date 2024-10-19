from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from getdata import load_stock_data

# Load dữ liệu
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

# Các tham số để thử nghiệm
activation_functions = ['relu', 'tanh', 'logistic']
hidden_layer_sizes = [(50, 50), (100, 50), (100, 100)]
solvers = ['adam', 'sgd']
max_iterations = [1000, 2000, 3000, 4000, 5000]
best_model = None
best_mse = float('inf')
best_params = {}

# Vòng lặp tìm kiếm mô hình tốt nhất
for activation in activation_functions:
    for hidden_layers in hidden_layer_sizes:
        for solver in solvers:
            for max_iter in max_iterations:
                print(f"Training model with activation={activation}, layers={hidden_layers}, solver={solver}")

                # Initialize and train the model
                mlp = MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    activation=activation,
                    solver=solver,
                    max_iter=max_iter,
                    random_state=42
                )
                mlp.fit(X_train_scaled, y_train)

                # Predict on the validation set
                y_val_pred = mlp.predict(X_val_scaled)

                # Calculate evaluation metrics
                mse = mean_squared_error(y_val, y_val_pred)
                mae = mean_absolute_error(y_val, y_val_pred)
                r2 = r2_score(y_val, y_val_pred)

                print(f"Validation MSE: {mse}, MAE: {mae}, R2: {r2}")

                # Track the best model based on MSE
                if mse < best_mse:
                    best_mse = mse
                    best_model = mlp
                    best_params = {
                        'activation': activation,
                        'hidden_layers': hidden_layers,
                        'solver': solver,
                        'max_iter': max_iter
                    }

# Đánh giá mô hình tốt nhất trên tập test
y_test_pred = best_model.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Best Model Test MSE: {test_mse}, MAE: {test_mae}, R2: {test_r2}")
print(f"Best Hyperparameters: {best_params}")

# Dự đoán cho các tập dữ liệu với mô hình tốt nhất
y_train_pred = best_model.predict(X_train_scaled)
y_val_pred = best_model.predict(X_val_scaled)
y_test_pred = best_model.predict(X_test_scaled)

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

