import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def add_noise_to_label(labels, noise_rate_0, noise_rate_1):
    idx_0 = np.where(labels == 0)[0]
    idx_1 = np.where(labels == 1)[0]
    num_noisy_data_0 = int(noise_rate_0 * len(idx_0))
    num_noisy_data_1 = int(noise_rate_1 * len(idx_1))
    idxs_0 = np.random.choice(idx_0, num_noisy_data_0, replace=False)
    idxs_1 = np.random.choice(idx_1, num_noisy_data_1, replace=False)
    labels[idxs_0] = 1 - labels[idxs_0]
    labels[idxs_1] = 1 - labels[idxs_1]
    return labels

datas, labels = make_classification(n_samples=9999, n_features=2, n_classes=2, n_redundant=0)

datas_train, datas_test, labels_train, labels_test = train_test_split(datas, labels, test_size=0.2)

plt.figure(1)
plt.title('Linearly Separable Synthetic Dataset: trainig data before introducing noise')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.scatter(datas_train[labels_train == 0, 0], datas_train[labels_train == 0, 1], label='class 0')
plt.scatter(datas_train[labels_train == 1, 0], datas_train[labels_train == 1, 1], marker='.', label='class 1')
plt.legend(loc='best')
plt.savefig('./Figure1.jpg')

labels_train = add_noise_to_label(labels_train, 0.4, 0.4)
# labels_train = add_noise_to_label(labels_train, 0.2, 0.2)
plt.figure(2)
plt.title('Linearly Separable Synthetic Dataset: trainig data after introducing noise')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.scatter(datas_train[labels_train == 0, 0], datas_train[labels_train == 0, 1], label='class 0')
plt.scatter(datas_train[labels_train == 1, 0], datas_train[labels_train == 1, 1], marker='.', label='class 1')
plt.legend(loc='best')
plt.savefig('./Figure2.jpg')

# Llogistic regression model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(datas_train, labels_train)
predictions = logistic_regression_model.predict(datas_test)
print("Logistic regression model(baseline model): ")
print("Accuracy: {:.2f}%".format(accuracy_score(labels_test, predictions) * 100))
plt.figure(3)
plt.title('Linearly Separable Synthetic Dataset: baseline model prediction')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.scatter(datas_test[predictions == 0, 0], datas_test[predictions == 0, 1], label='class 0')
plt.scatter(datas_test[predictions == 1, 0], datas_test[predictions == 1, 1], marker='.', label='class 1')
plt.legend(loc='best')
plt.savefig('./Figure3.jpg')

# SVM
svm_model = SVC()
svm_model.fit(datas_train, labels_train)
predictions = svm_model.predict(datas_test)
print("C-SVM with proposed method:")
print("Accuracy: {:.2f}%".format(accuracy_score(labels_test, predictions) * 100))
plt.figure(4)
plt.title('Linearly Separable Synthetic Dataset: model with proposed method prediction')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.scatter(datas_test[predictions == 0, 0], datas_test[predictions == 0, 1], label='class 0')
plt.scatter(datas_test[predictions == 1, 0], datas_test[predictions == 1, 1], marker='.', label='class 1')
plt.legend(loc='best')
plt.savefig('./Figure4.jpg')

breast_cancer_data = pd.read_csv("breast-cancer.csv")
breast_cancer_data = breast_cancer_data.drop("id", axis=1)
breast_cancer_data["diagnosis"] = breast_cancer_data["diagnosis"].replace({"M": 1, "B": 0})

real_labels = np.array(breast_cancer_data["diagnosis"])
datas = np.array(breast_cancer_data.drop("diagnosis", axis=1))

data_train, data_test, label_train, label_test = train_test_split(datas, real_labels, test_size=0.2)

label_train = add_noise_to_label(label_train, 0.4, 0.4)
# label_train = add_noise_to_label(label_train, 0.2, 0.2)
scaler = StandardScaler()
data_train_scaled = scaler.fit_transform(data_train)
data_test_scaled = scaler.transform(data_test)

# Logistic Regression
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(data_train_scaled, label_train)
predicted_labels = logistic_regression_model.predict(data_test_scaled)
print("Logistic regression model(baseline model): ")
print("Accuracy: {:.2f}%".format(accuracy_score(label_test, predicted_labels) * 100))

# SVM
svm_model = SVC()
svm_model.fit(data_train_scaled, label_train)
predicted_labels = svm_model.predict(data_test_scaled)
print("C-SVM with proposed method:")
print("Accuracy: {:.2f}%".format(accuracy_score(label_test, predicted_labels) * 100))
