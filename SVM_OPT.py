import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = r"C:\Users\katha\OneDrive\Desktop\ML Parameter Optimization\online_shoppers_intention.csv"
data = pd.read_csv(file_path)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nClass Distribution (Revenue):")
print(data['Revenue'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='Revenue', data=data, palette='viridis', legend=False)
plt.title("Class Distribution of Revenue")
plt.xlabel("Revenue (0: No Purchase, 1: Purchase)")
plt.ylabel("Count")
plt.show()

#categorical variables
data['Month'] = data['Month'].astype('category').cat.codes  # Convert month names to numbers
data['VisitorType'] = data['VisitorType'].astype('category').cat.codes  # Convert visitor type to numbers
data['Weekend'] = data['Weekend'].astype(int)  # Convert True/False to 1/0
data['Revenue'] = data['Revenue'].astype(int)  # Convert True/False to 1/0

#Correlation heatmap
numerical_features = data.select_dtypes(include=[np.number])
correlation_matrix = numerical_features.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

#Feature-distributions
plt.figure(figsize=(15, 10))
for i, column in enumerate(numerical_features.columns, 1):
    plt.subplot(5, 4, i)
    sns.histplot(data[column], kde=True, bins=30, color='blue')
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.tight_layout()
plt.show()

X = data.drop(columns=['Revenue'])
y = data['Revenue']

samples = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    samples.append((X_train, X_test, y_train, y_test))

param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

results = []
for i, (X_train, X_test, y_train, y_test) in enumerate(samples):
    print(f"Optimizing SVM for Sample S{i+1}...")
    
    grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', verbose=0)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    accuracy = accuracy_score(y_test, best_model.predict(X_test))
    
    results.append({
        "Sample": f"S{i+1}",
        "Best Accuracy": accuracy,
        "Best Parameters": best_params
    })

print("\n--- Results Summary ---")
for res in results:
    print(f"{res['Sample']}: Accuracy = {res['Best Accuracy']:.2f}, Parameters = {res['Best Parameters']}")

best_sample = max(results, key=lambda x: x['Best Accuracy'])

iterations = list(range(1, len(grid_search.cv_results_['mean_test_score']) + 1))
accuracies = grid_search.cv_results_['mean_test_score']

plt.figure(figsize=(8, 5))
plt.plot(iterations, accuracies, label=f"Sample {best_sample['Sample']} - Best Accuracy")
plt.title("Convergence Graph of SVM Optimization")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
