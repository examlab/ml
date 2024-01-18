import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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
clf=MultinomialNB()
clf.fit(Xtrain_dtm,ytrain)
MultinomialNB()
pred=clf.predict(Xtest_dtm)
print(pred)
print('Accuracy: ',accuracy_score(y,pred))
print('Confusion matrix: ',confusion_matrix(y,pred))
print('Precision: ',precision_score(y,pred))
print('Recall: ',recall_score(y,pred))


