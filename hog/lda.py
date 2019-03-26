from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from skimage_hog import HOG_Feature
import numpy as np
model = LDA()
model2 = LDA()
hog = HOG_Feature((96, 128), 9, 8, 2)
X = np.zeros((20, 5940))
for i in range(1, 12):
    X[i-1] = hog.vector(str(i) + ".JPG").round(3)
test = np.zeros(5940)
for i in range(11, 20):
    X[i] = test
Y = np.array([1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2])

model.fit(X, Y)
model.transform(X)
print(model.decision_function(np.array([test])))
print(model.predict_proba(np.array([test])))
print(model.predict(np.array([test])))

test2 = hog.vector("14.JPG")
print(model.decision_function(np.array([test2])))
print(model.predict_proba(np.array([test2])))
print(model.predict(np.array([test2])))

test3 = np.random.randint(0, 20, 5940)
print(model.decision_function(np.array([test3])))
print(float(model.predict_proba(np.array([test3]))[0][0]))
print(model.predict(np.array([test3])))