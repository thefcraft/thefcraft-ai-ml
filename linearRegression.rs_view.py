import struct
import numpy as np
from sklearn.datasets import make_regression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load(filename:str)->np.ndarray:
    with open(filename, 'rb') as f:
        buffer = f.read()
    # Read the size of the vector of f32 values
    size_bytes = buffer[:8]  # Assuming size is stored in the first 8 bytes as usize
    size = struct.unpack('<Q', size_bytes)[0]  # Read the usize value as little endian
    
    # Initialize list to store f32 values
    data = []
    # Iterate over the buffer to read f32 values
    for i in range(8, size*4+8, 4):
        f32_bytes = buffer[i:i+4]
        num = struct.unpack('<f', f32_bytes)[0]  # Read the f32 value as little endian
        data.append(num)
    assert size == len(data)
    shape = []
    for i in range(8 + size * 4, len(buffer), 8):
        usize_bytes = buffer[i:i+8]
        num = struct.unpack('<Q', usize_bytes)[0]  # Read the usize value as little endian
        shape.append(num)
    
    return np.array(data, dtype=np.float32).reshape(shape)
def save(arr:np.ndarray, filename:str):
    shape = arr.shape
    with open(filename, 'wb') as file:
        file.write(struct.pack('<Q', arr.size))
        for num in arr.reshape(-1):
            file.write(struct.pack('<f', num))
        for num in shape:
            file.write(struct.pack('<Q', num))

# X, Y = make_regression(n_samples=10000, n_features=400, noise=50)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# save(X_train, r"C:\ThefCraft\thefcraft-rust\nn-c\src\X_train.bin")
# save(Y_train, r"C:\ThefCraft\thefcraft-rust\nn-c\src\Y_train.bin")
# save(X_test, r"C:\ThefCraft\thefcraft-rust\nn-c\src\X_test.bin")
# save(Y_test, r"C:\ThefCraft\thefcraft-rust\nn-c\src\Y_test.bin")
# quit()

X_train = load(r"C:\ThefCraft\thefcraft-rust\nn-c\src\X_train.bin")
Y_train = load(r"C:\ThefCraft\thefcraft-rust\nn-c\src\Y_train.bin")
X_test = load(r"C:\ThefCraft\thefcraft-rust\nn-c\src\X_test.bin")
Y_test = load(r"C:\ThefCraft\thefcraft-rust\nn-c\src\Y_test.bin")

sklearn_model = linear_model.LinearRegression(fit_intercept=True)
sklearn_model.fit(X_train, Y_train)

sklearn_predictions = sklearn_model.predict(X_test)
custom_predictions = load(r"C:\ThefCraft\thefcraft-rust\nn-c\src\y_pred.bin")

mse_custom = mean_squared_error(Y_test, custom_predictions)
mse_sklearn = mean_squared_error(Y_test, sklearn_predictions)

r2_custom = r2_score(Y_test, custom_predictions)
r2_sklearn = r2_score(Y_test, sklearn_predictions)

print("Mean Squared Error (Custom Model):", mse_custom)
print("Mean Squared Error (Scikit-Learn Model):", mse_sklearn)
print("R-squared (Custom Model):", r2_custom)
print("R-squared (Scikit-Learn Model):", r2_sklearn)