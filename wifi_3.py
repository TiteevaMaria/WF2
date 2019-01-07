#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split # деление данных на тренировочные и тестовые
from sklearn import metrics, linear_model # подсчет точности, линейная регрессия (тренируемая модель)


# In[2]:



file = open("wifi_localization.txt") # исходные данные

WF = [] # качество сигнала для предсказания комнат 
RM = [] # номера комнат в соответствии с качеством сигнала WF

# первые семь значений строки - WF, сигнал, последнее - соответствующий ему номер комнаты RM
for s in file:
    WF.append(s.split()[:7])
    RM.append(s.split()[7:])

# преобразование в нампай 
WF = np.array(WF, dtype = int)
RM = np.array(RM, dtype = int)


# In[3]:


WF_train, WF_test, RM_train, RM_test = train_test_split(WF, RM, random_state = 0) # разбиение на тренировочные и тестовые данные


# In[4]:


logreg = linear_model.LogisticRegression(C = 10, solver = 'lbfgs', max_iter = 10000, multi_class = 'auto') # создание модели
logreg.fit(WF_train, RM_train) # тренировка модели


# In[5]:


pred = logreg.predict(WF_test) # отправка тестовых данных в модель
print("Accuracy:", metrics.accuracy_score(RM_test, pred)) # подсчет точности предсказания


# In[10]:


room = []
# сравнение предсказанного с действительным
for i, j in enumerate(RM_test): 
    if pred[i] != j:
        room.append(i)

spTest = []
# сохранение ошибочных предсказаний
for i in room: 
    spTest.append(WF_test[i])  
spTest = np.array(spTest)

# вывод
for i in room: 
    print("Должно быть:", RM_test[i], " Получилось:", pred[i])
print()
print("Вероятность, что сеть в комнате")
print("[        1              2             3             4        ]")
print(logreg.predict_proba(spTest))


# In[ ]:





# In[ ]:




