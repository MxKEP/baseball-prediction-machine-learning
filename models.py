# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 19:35:39 2022

@author: mxkep
"""

# Split the data
from sklearn.model_selection import train_test_split

y = svdata['events']
X = svdata.drop(['events', 'pitch_type', 'outs_when_up', 'release_pos_y'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,
                                                   random_state = 25)

#
# Standardize the data set
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# Create Decision Tree
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X_train_std, y_train)

# Feature Importance Plot

utl.pretty_importances_plot(
    model.feature_importances_, 
    [i for i in range(X_train.shape[1])],
    xlabel = 'Importance',
    ylabel = 'Feature',
    horizontal_label = 'Feature importance'
)


text_representation = tree.export_text(model)
with open("decision_tree.log", "w") as fout:
    fout.write(text_representation)


# Evaluate Decision Tree Model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

y_predict = model.predict(X_test_std)
y_pred_dt = y_predict





#
# Plot confusion matrix


pretty_conf_matrix(
    y_test = y_test,
    predictions = y_pred_dt,
    string = "Decision Tree")

print("\nDecision Tree ")
utl.cv_model(X_train_std, y_train, model, "Decision Tree")
print("Decision Tree Accuracy Score: ", accuracy_score(y_test, y_predict) * 100)
print("Confusion Matrix: \n", confusion_matrix(y_test, y_predict))
print("\n")

#####################################################################################

# Create Random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=70, random_state=123)
clf.fit(X_train_std, y_train)
y_predict = clf.predict(X_test_std)
y_pred_rf = y_predict


# Plot confusion matrix

pretty_conf_matrix(
    y_test = y_test,
    predictions = y_pred_rf,
    string = "Random Forest")

print("Random Forest ")
utl.cv_model(X_train_std, y_train, model, "Random Forest")
print("Random Forest Accuracy Score: ", accuracy_score(y_test, y_predict) * 100)
print("Confusion Matrix: \n", confusion_matrix(y_test, y_predict))
print("\n")





#######################################################################################
############## Train the Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)

# Evaluate model performance

y_predict = gnb.predict(X_test)
y_pred_nb = y_predict


# Plot confusion matrix

pretty_conf_matrix(
    y_test = y_test,
    predictions = y_pred_nb,
    string = "Naive Bayes")

print("Naive Bayes ")
utl.cv_model(X_train, y_train, model, "Naive Bayes")
print("Naive Bayes Classifier Accuracy Score: ", accuracy_score(y_test, y_predict)*100)
print("Confusion Matrix: \n", confusion_matrix(y_test, y_predict))
print("\n")


##################### Design SVM poly kernel  #####################


from sklearn import svm



clf = svm.SVC(kernel='rbf')

clf.fit(X_train_std, y_train)


y_predict = clf.predict(X_test_std)
y_pred_svm = y_predict



# Plot confusion matrix

pretty_conf_matrix(
    y_test = y_test,
    predictions = y_pred_svm,
    string = "SVM (rbf)")


print("SVM (polynomial) ")
utl.cv_model(X_train, y_train, clf, "SVM (polynomial)")
print("SVM (polynomial)  Accuracy Score: ", accuracy_score(y_test, y_pred_svm)*100)
print("SVM (polynomial) Confusion Matrix: \n", confusion_matrix(y_test, y_pred_svm))
print("\n")

###############################
###### Deep Neural Network ####
###############################

from tensorflow import keras
from tensorflow.keras.optimizers import Adam

model = keras.models.Sequential([
keras.layers.Dense(64, input_dim = 14, activation = 'relu'),
keras.layers.Dropout(0.4), # add Dropout layers to avoid overfitting
keras.layers.Dense(64, activation = 'relu'),
keras.layers.Dropout(0.4),
keras.layers.Dense(1)
])



y_pred_nn = model(X_train_std[:1]).numpy()
y_pred_nn


# Design and train model

optimizer = Adam(lr=0.001)


model.compile(optimizer=optimizer,
loss='binary_crossentropy',
metrics=['accuracy'])


#model.fit(X_train_std, y_train, epochs=15)

# Learning curve

hist = model.fit(X_train_std, y_train,  validation_split=0.2, epochs=50)

plt.subplot(2,1,1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Neural Network Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.subplot(2,1,2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Neural Network Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
plt.show()

#### For metrics

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

# fit the model
history = model.fit(X_train_std, y_train,  validation_split=0.2, epochs=50)

# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
####


##################

# Print tables
utl.print_table(y_test, y_pred_dt, y_pred_rf, y_pred_nb)
#utl.print_table_svm(y_test, y_pred_poly, y_pred_rbf, "SVM (polynomial)", "SVM (RBF)")


# Create dataframe for model accuracies
model_accuracies = [['DT', accuracy_score(y_test, y_pred_dt)*100], 
                    ['RF', accuracy_score(y_test, y_pred_rf)*100],
                    ['NB', accuracy_score(y_test, y_pred_nb)*100],
                    ['SVM',accuracy_score(y_test, y_pred_svm)*100],
                    ['NN', 62]]

df_ma = pd.DataFrame(model_accuracies, columns = ['Model', 'Accuracy'])


sns.catplot(x="Model", 
                        y="Accuracy", 
                        hue="Model", 
                        data=df_ma, 
                        palette="colorblind",
                        kind='bar',
                        height = 5,
                        aspect = 1.5,
                        legend=False).set(title='Model Accuracy')
 
 
 
 
 
 
 
 
 