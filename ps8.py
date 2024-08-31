import numpy as np
import scipy
import numpy
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode


# read in data
MNIST = scipy.io.loadmat(r'input/HW8_data1.mat')

# define X and y
X = MNIST['X']
y = MNIST['y']

# combine and shuffle
data = np.hstack((X, y))
np.random.shuffle(data)

# redefine X and y (shuffled together)
X = data[:, :-1]
y = data[:, -1]

# 1a. pick first 25 images to print
selected = data[:25]

# print 25 images with their label
fig, axs = plt.subplots(5, 5)
for i in range(25):
    ax = axs[i // 5, i % 5]
    img_label = selected[i, -1]

    img = np.reshape(selected[i, :-1], (20, 20)).T

    ax.imshow(img, cmap='gray')

    # for consistency
    if img_label == 10: img_label = 0.0

    ax.set_title(f"{img_label}", fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig(r'output/ps8-1-a-1.png')
plt.close()

# 1b. Split data
X_train = X[:4500, :]
X_test = X[4500:, :]
y_train = y[:4500]
y_test = y[4500:]

train_data = np.hstack((X_train, y_train.reshape(-1, 1)))

# 1c. Create 5 bootstapped samples
bootstrap = np.random.choice(train_data.shape[1], 1000, replace=True)
sample = train_data[bootstrap]
X1_dict = {
    'X': sample[:, :-1],
    'y': sample[:, -1]
}

bootstrap = np.random.choice(train_data.shape[1], 1000, replace=True)
sample = train_data[bootstrap]
X2_dict = {
    'X': sample[:, :-1],
    'y': sample[:, -1]
}

bootstrap = np.random.choice(train_data.shape[1], 1000, replace=True)
sample = train_data[bootstrap]
X3_dict = {
    'X': sample[:, :-1],
    'y': sample[:, -1]
}

bootstrap = np.random.choice(train_data.shape[1], 1000, replace=True)
sample = train_data[bootstrap]
X4_dict = {
    'X': sample[:, :-1],
    'y': sample[:, -1]
}

bootstrap = np.random.choice(train_data.shape[1], 1000, replace=True)
sample = train_data[bootstrap]
X5_dict = {
    'X': sample[:, :-1],
    'y': sample[:, -1]
}

# save to input folder
scipy.io.savemat(r'input/X1.mat', X1_dict)
scipy.io.savemat(r'input/X2.mat', X2_dict)
scipy.io.savemat(r'input/X3.mat', X3_dict)
scipy.io.savemat(r'input/X4.mat', X4_dict)
scipy.io.savemat(r'input/X5.mat', X5_dict)

from sklearn.metrics import accuracy_score

# 1d. train OvA SVM
SVM = SVC(kernel='rbf', decision_function_shape='ovr')
SVM.fit(X1_dict['X'], X1_dict['y'])
print("Support Vector Machine")

# i. error on training set
prediction = SVM.predict(X1_dict['X'])
print("Error on training set: ", 1 - accuracy_score(X1_dict['y'], prediction))

# ii. error on other training set
prediction = SVM.predict(X2_dict['X'])
print("Error on X2: ", 1 - accuracy_score(X2_dict['y'], prediction))
prediction = SVM.predict(X3_dict['X'])
print("Error on X3: ", 1 - accuracy_score(X3_dict['y'], prediction))
prediction = SVM.predict(X4_dict['X'])
print("Error on X4: ", 1 - accuracy_score(X4_dict['y'], prediction))
prediction = SVM.predict(X5_dict['X'])
print("Error on X5: ", 1 - accuracy_score(X5_dict['y'], prediction))

# iii. error on testing set
prediction1 = SVM.predict(X_test)
print("Error on testing set: ", 1 - accuracy_score(y_test, prediction1))

# 1e. KNN (K = 5)
KNN = KNeighborsClassifier(5)
KNN.fit(X2_dict['X'], X2_dict['y'])
print("\nKNN (k = 5)")

# i. error on training set
prediction = KNN.predict(X2_dict['X'])
print("Error on training set: ", 1 - accuracy_score(X2_dict['y'], prediction))

# ii. error on other training set
prediction = KNN.predict(X1_dict['X'])
print("Error on X1: ", 1 - accuracy_score(X1_dict['y'], prediction))
prediction = KNN.predict(X3_dict['X'])
print("Error on X3: ", 1 - accuracy_score(X3_dict['y'], prediction))
prediction = KNN.predict(X4_dict['X'])
print("Error on X4: ", 1 - accuracy_score(X4_dict['y'], prediction))
prediction = KNN.predict(X5_dict['X'])
print("Error on X5: ", 1 - accuracy_score(X5_dict['y'], prediction))

# iii. error on testing set
prediction2 = KNN.predict(X_test)
print("Error on testing set: ", 1 - accuracy_score(y_test, prediction2))

# 1f. Logistic
Logistic = LogisticRegression()
Logistic.fit(X3_dict['X'], X3_dict['y'])
print("\nLogistic Regression")

# i. error on training set
prediction = Logistic.predict(X3_dict['X'])
print("Error on training set: ", 1 - accuracy_score(X3_dict['y'], prediction))

# ii. error on other training sets
prediction = Logistic.predict(X1_dict['X'])
print("Error on X1: ", 1 - accuracy_score(X1_dict['y'], prediction))
prediction = Logistic.predict(X2_dict['X'])
print("Error on X2: ", 1 - accuracy_score(X2_dict['y'], prediction))
prediction = Logistic.predict(X4_dict['X'])
print("Error on X4: ", 1 - accuracy_score(X4_dict['y'], prediction))
prediction = Logistic.predict(X5_dict['X'])
print("Error on X5: ", 1 - accuracy_score(X5_dict['y'], prediction))

# iii. error on testing set
prediction3 = Logistic.predict(X_test)
print("Error on testing set: ", 1 - accuracy_score(y_test, prediction3))

# 1g. Decision Tree
DT = DecisionTreeClassifier()
DT.fit(X4_dict['X'], X4_dict['y'])
print("\nDecision Tree Classifier")

# i. error on training set
prediction = DT.predict(X4_dict['X'])
print("Error on training set: ", 1 - accuracy_score(X4_dict['y'], prediction))

# ii. error on other training sets
prediction = DT.predict(X1_dict['X'])
print("Error on X1: ", 1 - accuracy_score(X1_dict['y'], prediction))
prediction = DT.predict(X2_dict['X'])
print("Error on X2: ", 1 - accuracy_score(X2_dict['y'], prediction))
prediction = DT.predict(X3_dict['X'])
print("Error on X3: ", 1 - accuracy_score(X3_dict['y'], prediction))
prediction = DT.predict(X5_dict['X'])
print("Error on X5: ", 1 - accuracy_score(X5_dict['y'], prediction))

# iii. error on testing set
prediction4 = DT.predict(X_test)
print("Error on testing set: ", 1 - accuracy_score(y_test, prediction4))

# 1h.
RF = RandomForestClassifier(n_estimators=85)
RF.fit(X5_dict['X'], X5_dict['y'])
print("\nRandom Forest Classifier")

# i. error on training set
prediction = RF.predict(X5_dict['X'])
print("Error on training set: ", 1 - accuracy_score(X5_dict['y'], prediction))

# ii. error on other training sets
prediction = RF.predict(X1_dict['X'])
print("Error on X1: ", 1 - accuracy_score(X1_dict['y'], prediction))
prediction = RF.predict(X2_dict['X'])
print("Error on X2: ", 1 - accuracy_score(X2_dict['y'], prediction))
prediction = RF.predict(X3_dict['X'])
print("Error on X3: ", 1 - accuracy_score(X3_dict['y'], prediction))
prediction = RF.predict(X4_dict['X'])
print("Error on X4: ", 1 - accuracy_score(X4_dict['y'], prediction))

# iii. error on testing set
prediction5 = RF.predict(X_test)
print("Error on testing set: ", 1 - accuracy_score(y_test, prediction5))

# 1i. Use majority vote and report accuracy
predictions = np.vstack((prediction1, prediction2, prediction3, prediction4, prediction5))
mode, count = mode(predictions) # use mode to find most common prediction


print("\nError on majority vote: ", 1 - accuracy_score(y_test, mode))