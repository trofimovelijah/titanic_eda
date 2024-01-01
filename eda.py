from catboost.datasets import titanic

if __name__ == '__main__':
    titanic_train, _ = titanic()

    titanic_train.fillna(-999, inplace=True)
    preprocessed = titanic_train.drop(['PassengerId'], axis=1)
    preprocessed.to_csv('preprocessed_data.csv', index=False)
