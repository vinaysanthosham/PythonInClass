from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
# load the iris datasets
dataset = datasets.load_iris()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


model = GaussianNB()
model.fit(x_train, y_train)
print(model.score(x_test,y_test))