import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error

def select_features_univariate(X, y, k):
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return selected_features

def evaluate_performance(X, y, selected_features):
    mae_values = []
    for k in range(1, len(selected_features) + 1):
        selected_X = X[selected_features[:k]]
        X_train, X_test, y_train, y_test = train_test_split(selected_X, y, test_size=0.2, random_state=1)
        model = LinearSVR()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mae_values.append(mae)
    return mae_values

# Assuming you have X and y defined

input_k = 10  # Maximum value of k
selected_features_list = []
mae_values_list = []

for k in range(1, input_k + 1):
    selected_features = select_features_univariate(X, y, k)
    selected_features_list.append(selected_features)
    mae_values = evaluate_performance(X, y, selected_features)
    mae_values_list.append(mae_values)

# Plotting the MAE with different numbers of selected features
for k, mae_values in zip(range(1, input_k + 1), mae_values_list):
    plt.plot(range(1, k + 1), mae_values, marker='o', label=f'k={k}')

plt.xlabel('Number of Features')
plt.ylabel('Mean Absolute Error')
plt.title('MAE with Different Numbers of Selected Features')
plt.legend()
plt.show()
