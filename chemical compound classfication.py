import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

dataset=pd.read_csv("musk.csv")
x=dataset.iloc[:,3:169].values
y=dataset.iloc[:,-1].values



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)




import tensorflow
import keras
from tensorflow.python.keras.models import Sequential
from  tensorflow.python.keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(83,kernel_initializer="uniform",activation='relu',input_dim=166))
classifier.add(Dense(83,kernel_initializer="uniform",activation='relu'))
classifier.add(Dense(83,kernel_initializer="uniform",activation='relu'))
classifier.add(Dense(1,kernel_initializer="uniform",activation='sigmoid',input_dim=166))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
r=classifier.fit(x_train,y_train,batch_size=10,epochs=80,validation_data=(x_val, y_val))

print(r.history.keys())

plt.plot(r.history['accuracy'])
plt.plot(r.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()
y_pred=classifier.predict(x_test)
y_pred=y_pred[:,0]
y_classes = classifier.predict_classes(x_test, verbose=0)
y_classes = y_classes[:, 0]
precision = precision_score(y_test, y_classes)

recall = recall_score(y_test, y_classes)

score = f1_score(y_classes, y_test)

cm=confusion_matrix(y_test,y_classes)
ac=accuracy_score(y_test,y_classes)


 









