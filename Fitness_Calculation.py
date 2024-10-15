from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
def wrapper_KNN(feat,label):
    k=3;
    X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=12345)
      
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    train_preds = knn_model.predict(X_train)
    mse = mean_squared_error(y_train, train_preds)
    rmse = sqrt(mse)
    return rmse
def mean_squared_error(actual, predicted):
	sum_square_error = 0.0
	for i in range(len(actual)):
		sum_square_error += (actual[i] - predicted[i])**2.0
	mean_square_error = 1.0 / len(actual) * sum_square_error
	return mean_square_error
def Fitness_function(selfea,label,X,ws):
    
    # calculate mean squared error

    ws = [0.99, 0.01];
    error    = wrapper_KNN(selfea,label);
    
    # % Number of selected features
    num_feat = len(sum(X == 1));
    
    # % Total number of features
    max_feat = len(X); 
    
    # % Set alpha & beta
    alpha    = ws[0]; 
    beta     = ws[1];
    # % Cost function 
    cost     = alpha * error + beta * (num_feat / max_feat);
    
    # cost = cost[0]
    return cost