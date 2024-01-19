import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix


x=np.array([3.78,2.44,2.09,0.14,1.72,1.65,4.92,4.37,4.96,4.52,3.69,5.88]).reshape(-1,1)
y=np.array([0,0,0,0,0,0,1,1,1,1,1,1])


model=LogisticRegression(solver='liblinear',random_state=0)
model.fit(x,y)


pred=model.predict(x)
print("Predicted values are:",pred)
print("Actual Values are:",y)


print('Accuracy: ',accuracy_score(y,pred))
print('Confusion matrix: ',confusion_matrix(y,pred))
print('Precision: ',precision_score(y,pred))
print('Recall: ',recall_score(y,pred))
