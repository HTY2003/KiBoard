from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np

class WeakLearner:
    def __init__(self):
        self.model = LDA(n_components=2)

    def train(self, correct_data, wrong_data):
        print(correct_data.shape)
        X = np.concatenate((correct_data, wrong_data))
        Y = np.concatenate((np.zeros(len(correct_data)), np.ones(len(wrong_data))))
        print(X.shape, Y.shape)
        self.model.fit(X, Y)

    def predict(self, data, details=False):
        if details:
            print(self.model.decision_function(data))
            print(self.model.predict_proba(data))
            print(self.model.predict(data))
        return self.model.predict(data) == 1

    def test(self, test_data, answers):
        return self.model.score(test_data, answers)