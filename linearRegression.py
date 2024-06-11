import numpy as np
import torch # for linearRegressioncuda

class linearRegression:
    def __init__(self, fit_intercept=True):
        self.M:np.ndarray = None
        self.b:np.ndarray = None
        self.fit_intercept = fit_intercept
    
    def fit(self, X:np.ndarray, Y:np.ndarray):
        if self.fit_intercept == True:
            # TODO: use y = mx for solving the problem not y = mx+b
            ...
            
        n=len(X)   
        avg_y = Y.mean()
        avg_yxj = np.mean(Y[:, np.newaxis] * X, axis=0)  # avg_yxj = np.array([(Y*X[:, i]).mean() for i in range(n_features)]) 
        avg_xj = X.mean(axis=0)
        avg_xjxj = np.dot(X.T, X) / n    # avg_xjxj = np.sum([np.outer(X[i], X[i])/n for i in range(n)], axis=0)

        A = n * np.outer(avg_xj, avg_xj) - avg_xjxj # A = np.array([[n*avg_xj[jj]*avg_xj[j] - avg_xjxj[jj][j] for jj in range(n_features)] for j in range(n_features)])
        B = n * avg_y * avg_xj - avg_yxj  # B = np.array([n*avg_y*avg_xj[j]-avg_yxj[j] for j in range(n_features)])

        # M = np.linalg.solve(A, B)
        self.M, _, _, _ = np.linalg.lstsq(A, B, rcond=None) # Numerical Stability: For numerical stability, it's a good idea to use np.linalg.lstsq instead of directly solving the system of equations. This function is better suited for solving over-determined systems, and it provides a more stable solution, especially when the matrix is ill-conditioned.
        self.b = avg_y - np.dot(self.M, avg_xj)  # b = avg_y - np.array([M[j]*avg_xj[j] for j in range(n_features)]).sum()
        
    def predict(self,X:np.ndarray):
        # return (X*self.M).sum(axis=1)+self.b
        return np.dot(X, self.M) + self.b
    
    def __repr__(self) -> str:
        eq = ""
        for idx, i in enumerate(self.M):
            eq += f"{i} x{idx+1} + "
        eq += f"({self.b})"
        return eq

class linearRegressioncuda: # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, fit_intercept=True):
        self.M:torch.Tensor = None
        self.b:torch.Tensor = None
        self.fit_intercept = fit_intercept
    
    def fit(self, X:torch.Tensor, Y:torch.Tensor):
        with torch.no_grad():
            n=len(X)   
            avg_y = Y.mean()
            avg_yxj = (Y.unsqueeze(dim=1) * X).mean(dim=0)
            avg_xj = X.mean(dim=0)
            avg_xjxj = torch.mm(X.T, X) / n

            A = n * torch.outer(avg_xj, avg_xj) - avg_xjxj
            B = n * avg_y * avg_xj - avg_yxj

            # least_squares = lambda A, B: torch.matmul(torch.linalg.pinv(A), B)
            self.M = torch.matmul(torch.linalg.pinv(A), B)
            self.b = avg_y - self.M.dot(avg_xj)
        
    def predict(self,X:torch.Tensor):
        # return (X*self.M).sum(axis=1)+self.b
        with torch.no_grad():
            return torch.matmul(X, self.M) + self.b

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    
    X, Y = make_regression(n_samples=100000, n_features=100, noise=50)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


    custom_model_time_t = time.time()
    custom_model = linearRegression()
    custom_model.fit(X_train, Y_train)
    custom_model_time_t = time.time() - custom_model_time_t

    sklearn_model_time_t = time.time()
    sklearn_model = linear_model.LinearRegression(fit_intercept=True)
    sklearn_model.fit(X_train, Y_train)
    sklearn_model_time_t = time.time() - sklearn_model_time_t

    custom_model_time_i = time.time()
    custom_predictions = custom_model.predict(X_test)
    custom_model_time_i = time.time() - custom_model_time_i

    sklearn_model_time_i = time.time()
    sklearn_predictions = sklearn_model.predict(X_test)
    sklearn_model_time_i = time.time() - sklearn_model_time_i


    mse_custom = mean_squared_error(Y_test, custom_predictions)
    mse_sklearn = mean_squared_error(Y_test, sklearn_predictions)

    r2_custom = r2_score(Y_test, custom_predictions)
    r2_sklearn = r2_score(Y_test, sklearn_predictions)

    print("Mean Squared Error (Custom Model):", mse_custom)
    print("Mean Squared Error (Scikit-Learn Model):", mse_sklearn)
    print("R-squared (Custom Model):", r2_custom)
    print("R-squared (Scikit-Learn Model):", r2_sklearn)
    print("(Custom Model):", custom_model_time_t)
    print("(Scikit-Learn Model):", sklearn_model_time_t)
    print("(Custom Model) inf:", custom_model_time_i)
    print("(Scikit-Learn Model) inf:", sklearn_model_time_i)
    
    custom_model_time_t = max(custom_model_time_t,0.00000000000000000000000000000001)
    custom_model_time_i = max(custom_model_time_i,0.00000000000000000000000000000001)
    print(sklearn_model_time_t/custom_model_time_t, "times Faster in training.")
    print((sklearn_model_time_i+sklearn_model_time_t)/(custom_model_time_i+custom_model_time_t), "times overall Faster.")
    
    # Plot
    plt.figure(figsize=(12, 6))

    # Custom Model Predictions
    plt.subplot(1, 2, 1)
    plt.scatter(Y_test, custom_predictions, color='blue', alpha=0.5)
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], '--', color='red', linewidth=2)
    plt.title('Custom Model Predictions')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

    # Scikit-Learn Model Predictions
    plt.subplot(1, 2, 2)
    plt.scatter(Y_test, sklearn_predictions, color='orange', alpha=0.5)
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], '--', color='red', linewidth=2)
    plt.title('Scikit-Learn Model Predictions')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

    plt.tight_layout()
    
    # Data
    models = ['Custom Model', 'Scikit-Learn Model']
    times_train = [custom_model_time_t, sklearn_model_time_t]
    times_inf = [custom_model_time_i, sklearn_model_time_i]
    
    # Plot
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.bar(models, times_train, color=['blue', 'orange'])
    plt.title('Training Time')
    plt.ylabel('Time (s)')

    plt.subplot(1, 2, 2)
    plt.bar(models, times_inf, color=['blue', 'orange'])
    plt.title('Inference Time')
    plt.ylabel('Time (s)')

    plt.tight_layout()
    plt.show()
