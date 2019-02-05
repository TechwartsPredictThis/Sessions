import pandas as pd 
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from ann_visualizer.visualize import ann_viz
import matplotlib.pyplot as plt
from sklearn import datasets

#plt.style('ggplot')

seed = 5 # Seed indicates the intial state of randomization, if we fixt its vale, we will always get the same model
np.random.seed(5)

dataset = datasets.load_iris()

#iris_csv = pd.read_csv('/home/yash/python/ML/datasets/Iris.csv') # a dataframe object
#print(iris_csv)
iris_dataset = iris_csv.values
X = dataset[:,:4]
Y = dataset[:,4]
#print(Y)
"""
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(Y)
Y_encoded = label_encoder.transform(Y)
#print(Y_encoded)
"""
Y_one_hot = np_utils.to_categorical(Y)
#print(Y_one_hot)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y_one_hot,test_size=0.33,random_state=seed)
#print(Y_train)
model = Sequential()
model.add(Dense(8,activation='relu',input_dim=4)) # we need to specify the number of inputs per neuron for mostly the first layer 
model.add(Dense(3,activation='softmax'))  # the other layers implicitly understand the number of inputs per neuron

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
trained = model.fit(X_train,Y_train,epochs=200,batch_size=10,verbose=0) # verbose=0 switches off debugging
# trained = model.fit(X, Y, validation_split=0.33, epochs=200, batch_size=10, verbose=0) # another method of cross validation
print("Trained model accuracy %2.3f"%(trained.history['acc'][199]*100)) # accuracy at the last epoch


plt.plot(trained.history['acc'])
#plt.plot(trained.history['loss'])  
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


score = model.evaluate(X_test,Y_test,batch_size=10,verbose=0)
print("loss : %.3f  accuracy : %0.3f"%(score[0],score[1]*100))
prediction = model.predict(X_test)
print("1st prediction : "+str(prediction[0][0])+" "+str(prediction[0][1])+" "+str(prediction[0][2]))
print("%1.3f"%sum(prediction[0]))

#plot_model(model,'model1.png')
ann_viz(model,view=True,filename='model1.gv',title='iris_classfication_ann') # creating an image of the model that we created 