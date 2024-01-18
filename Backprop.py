import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score



msg=pd.read_csv('/content/document.csv',names=['message','label'])
print("Total instance of data set:",m.shape[0])
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
y=msg.labelnum
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y)
count_v=CountVectorizer()


Xtrain_dtm=count_v.fit_transform(Xtrain)
Xtest_dtm=count_v.fit(transform(Xtest))
df=pd.DataFrame(Xtrain_dtm.toarray(),columns=count_v.get_feature_names_out())
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(Xtrain_dm, ytrain)


pred = clf.predict(Xtest_dm)
print(pred)


print('Accuracy: ',accuracy_score(y,pred))
print('Confusion matrix: ',confusion_matrix(y,pred))
print('Precision: ',precision_score(y,pred))
print('Recall: ',recall_score(y,pred))


