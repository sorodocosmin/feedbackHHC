from sklearn.ensemble import RandomForestClassifier


class Random_Forest_Classifier:
    def __init__(self, x_train, y_train):
        self.__model = RandomForestClassifier()
        self.__x_train = x_train
        self.__y_train = y_train

    def train(self):
        self.__model.fit(self.__x_train, self.__y_train)

    def predict_instance(self, test):
        return self.__model.predict(test)
