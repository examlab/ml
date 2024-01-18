import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
m=pd.read_csv('document.csv',names=['messagge','label'])
print("Total instance of data set:",m.shape[0])
m['labelnum']=ms.label.map({'pos':1,'neg':0})
x=m.message
y=m.labelnum
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y)
count_v=coutVectorizer()
Xtrain_dtm=count_v.fit_transform(Xtrain)
Xtest_dtm=count_v.fit(transform(Xtest))
df=pd.DataFrame(Xtrain_dtm.toarray(),columns=count_v.get_feature_names_out())
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(Xtrain_dm, ytrain)
pred = clf.predict(Xtest_dm)
print(pred)
print(accuracy_score(ytest,pred))
print(confusion_matrix(ytest,pred))
print(precision_score(ytest,pred))
print(recall_score(ytest,pred))


