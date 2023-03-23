import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix  # tahmin degerlendime
# Earlystopping ve dropoutu ozellikle overfitting problemini cozmek için kullanacagız
dataFrame = pd.read_excel("maliciousornot.xlsx")
print(dataFrame)
print(dataFrame.info())
print(dataFrame.describe())
print(dataFrame.corr()["Type"].sort_values())
# Sayma grafigi
sbn.countplot(x="Type", data=dataFrame)
plt.show()
# corelasyonu grafik sekilde gormek icin asgidaki kodu kullan
dataFrame.corr()["Type"].sort_values().plot(kind="bar")
plt.show()

y = dataFrame["Type"].values
x = dataFrame.drop("Type", axis=1).values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=15)
# Butun train verileri 0 ve 1 arasında kucultuluyor
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)
# model = Sequential()
# # Unitsi 30 vermemizin nedeni 30 kolonumuz var esit olması oneriliyor
# model.add(Dense(units=30, activation="relu"))
# model.add(Dense(units=15, activation="relu"))
# model.add(Dense(units=15, activation="relu"))
# model.add(Dense(units=1, activation="sigmoid"))
# # Normalde lossu mse koyuyorduk fakat classification ornegi oldugu için degistirdik
# model.compile(loss="binary_crossentropy", optimizer="adam")

# model.fit(x=x_train, y=y_train, epochs=700,
#           validation_data=(x_test, y_test), verbose=1)

# modelKaybi = pd.DataFrame(model.history.history)
# modelKaybi.plot()
# plt.show()
# Model istenilenden farklı hareket etmeye baslarsa epochu durdurmak lazım bunun içinde
# early stopping gerekiyor

# model = Sequential()
# # Unitsi 30 vermemizin nedeni 30 kolonumuz var esit olması oneriliyor
# model.add(Dense(units=30, activation="relu"))
# model.add(Dense(units=15, activation="relu"))
# model.add(Dense(units=15, activation="relu"))
# model.add(Dense(units=1, activation="sigmoid"))
# # Normalde lossu mse koyuyorduk fakat classification ornegi oldugu için degistirdik
# model.compile(loss="binary_crossentropy", optimizer="adam")

earlyStopping = EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=25)
# model.fit(x=x_train, y=y_train, epochs=700,
#           validation_data=(x_test, y_test), verbose=1, callbacks=[earlyStopping])
# # Earlystopping validationun artmaya basladıgı zaömanepochs degerini durduruyor
# # buda yetersiz geliyorsa dropout kullanılmalı
# modelKaybi = pd.DataFrame(model.history.history)
# modelKaybi.plot()
# plt.show()

model = Sequential()
# Dropout degerini yukseltince iyilesiyor fakat belli bi degerin ustune cıkınca patlıyor
model.add(Dense(units=30, activation="relu"))
model.add(Dropout(0.7))

model.add(Dense(units=15, activation="relu"))
model.add(Dropout(0.7))

model.add(Dense(units=15, activation="relu"))
model.add(Dropout(0.7))

model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")

model.fit(x=x_train, y=y_train, epochs=700,
          validation_data=(x_test, y_test), verbose=1, callbacks=[earlyStopping])

kayipDf = pd.DataFrame(model.history.history)
kayipDf.plot()
plt.show()

tahminlerimiz = np.argmax(model.predict(x_test), axis=1)
print(tahminlerimiz)


print(classification_report(y_test, tahminlerimiz))

print(confusion_matrix(y_test, tahminlerimiz))
