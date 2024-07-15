import numpy as np
import cv2
import warnings
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

# loading the dataset
digits = load_digits()

X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier()

# best parameters obtained using GridSearchCV
clf = KNeighborsClassifier(metric="manhattan",n_neighbors=5,weights="distance")

# average accuracy after cross validation
accuracy = cross_val_score(clf, X_scaled, np.ravel(y), cv=10, scoring="accuracy")

print("ACCURACY:",accuracy.mean())

# fitting the model on entire dataset
clf.fit(X_scaled, y)


def preprocessing(point_list):
    """
    :param point_list: list of points (x,y coordinates)
    :return: preprocessed image-like matrix
    """
    number = np.zeros((240, 240))
    for cord in point_list:
        if cord[1] and cord[0] >= 0:
            cord[1] = cord[1] - 10
            cord[0] = cord[0] - 10
            number[cord[1], cord[0]] = 1
        else:
            print("error")
            break

    number = cv2.flip(number, 1)
    kernel = np.ones((5, 5))
    number = cv2.dilate(number, kernel, iterations=5)

    number = number * 16

    number = cv2.resize(number, (8, 8), interpolation=cv2.INTER_AREA)
    number = scaler.fit_transform(number.reshape(-1, 1))
    number = number.reshape(64)

    return number


def predict(data):
    """
    Function used for number classification
    :param data: image-like matrix
    :return: detected number
    """
    return int(clf.predict([data]))
