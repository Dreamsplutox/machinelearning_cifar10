#!/usr/bin/env python
# coding: utf-8

# ### Démarrage de tensorboard et imports principaux

# In[49]:


# Démarrage de tensorboard pour notebook
get_ipython().run_line_magic('load_ext', 'tensorboard')

import sys
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
import datetime
import os
import pandas as pd
from PIL import Image
import shutil  

from tensorflow.python.framework import ops #pour tenter de reset tensorboard, sans grand succès
ops.reset_default_graph()


# ### Fonctions pour préparer le dataset

# In[50]:


# Chargement des datasets de train et de validation + one hot encoding
def load_dataset():
    # Chargement des données cifar10
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode encoding sur les labels
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# Normalisation pour accroître la vitesse du modèle (en redimensionnant les pixels)
def prep_pixels(train, test):
    # Convertion des int en float
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # Normalisation pour avoir des nombres entre 0 et 1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # Retourner les images normalisées
    return train_norm, test_norm


# ### Classe Python pour définir les individus

# In[74]:


# Classe pour les convnets
class IndividuConvnets:
    def __init__(self, indiv_id='1', epochs=10, nb_layers=10, l1=0, l2=0, dropout=0, filters_per_layers=64, activation='relu', kernel=(3,3), padding='same', max_pool=0):
        # Initialisation de nos variables
        self.time_fit = datetime.datetime.now()
        self.my_reguralizer = 'he_uniform'
        
        if nb_layers <= 2:
            nb_layers = 2
            
        self.loss = 0
        self.accuracy = 0
        self.indiv_id = indiv_id
        self.epochs = epochs
        self.nb_layers = nb_layers

        # on ne met pas les 2 actifs ==> à vérifier ==> On peut mettre les 2 mais pour l'instant on le gère pas
        if (l1 != 0 and l2 != 0): 
            self.l1 = 0
            self.l2 = 0
        else:
            self.l1 = l1
            self.l2 = l2

        self.dropout = dropout
        self.filters_per_layers = filters_per_layers
        self.activation = activation
        self.kernel = kernel
        self.padding = padding
        self.max_pool = max_pool
    
    # TO STRING (retourne une liste avec tous les attributs utiles de la classe)
    def __str__(self):
        ma_liste = []
        ma_liste.append("indiv_id:{},\n ".format(self.indiv_id))
        ma_liste.append("epochs:{},\n ".format(self.epochs))
        ma_liste.append("nb_layers:{},\n ".format(self.nb_layers))
        ma_liste.append("l1:{},\n ".format(self.l1))
        ma_liste.append("l2:{},\n ".format(self.l2))
        ma_liste.append("dropout:{},\n ".format(self.dropout))
        ma_liste.append("filters_per_layers:{},\n ".format(self.filters_per_layers))
        ma_liste.append("activation:{},\n ".format(self.activation))
        ma_liste.append("kernel:\n ")
        ma_liste.append("{},\n ".format(self.kernel))
        ma_liste.append("padding:{},\n ".format(self.padding))
        ma_liste.append("maxpool:{}\n".format(self.max_pool))
            
        return ma_liste
    
    def create_and_train_model(self, trainX, trainY, testX, testY, main_directory):
        start = datetime.datetime.now()
        
        # Update indiv_id pour avoir un vrai ID unique
        self.indiv_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Choix d'un emplacement pour les logs
        log_dir=main_directory+"\\logs_"+self.indiv_id+"\\tensorboard_data\\"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        print("log dir = ",log_dir)
        
        #ON VA GERER L1 ET L2, A L AVENIR IL FAUDRAIT MIEUX LES GERER DANS UNE FONCTION ?
        if self.l1 > 0:
            self.my_regularizer = regularizers.l1(self.l1 / self.nb_layers)
        elif self.l2 > 0:
            self.my_regularizer = regularizers.l2(self.l2 / self.nb_layers)
        else:
            self.my_reguralizer = 'he_uniform'
        
        # Definir notre modèle basique, 2 couches de conv et 1 pool avant tout
        model = Sequential()
        model.add(Conv2D(self.filters_per_layers, self.kernel, activation=self.activation, kernel_initializer=self.my_reguralizer, padding=self.padding, input_shape=(32, 32, 3)))
        model.add(Conv2D(self.filters_per_layers, self.kernel, activation=self.activation, kernel_initializer=self.my_reguralizer, padding=self.padding))
        model.add(MaxPooling2D((2, 2)))
        
        # Faire toutes les convs nécessaires
        if self.nb_layers > 2:
            for i in range(2, self.nb_layers):
                if self.nb_layers - i != 1:
                    print("i = ", i)
                    # 2 conv + pool
                    model.add(Conv2D(self.filters_per_layers, self.kernel, activation=self.activation, kernel_initializer=self.my_reguralizer, padding=self.padding))
                    model.add(Conv2D(self.filters_per_layers, self.kernel, activation=self.activation, kernel_initializer=self.my_reguralizer, padding=self.padding))
                    model.add(MaxPooling2D((2, 2)))
                else:
                    # 1 conv + pool si nombre impair de couches (nb_layers)
                    model.add(Conv2D(self.filters_per_layers, self.kernel, activation=self.activation, kernel_initializer=self.my_reguralizer, padding=self.padding))
                    model.add(MaxPooling2D((2, 2)))
        
        
        # Fin des convs -> neural network classique (je n'utilise pas self.activation car ce n'est pas relié a ce neural net)
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer=self.my_reguralizer))
        model.add(Dense(10, activation='softmax'))

        # Compiler le modele
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        #save model png
        plot_model(model, "model.png")
        
        # Entrainer le modele
        history = model.fit(trainX, trainY, epochs=self.epochs, batch_size=64, validation_data=(testX, testY), verbose=0, callbacks=[tensorboard_callback])
        
        end = datetime.datetime.now()
        self.time_fit = end - start # J'arrive pas trop à le round, ça pourrait servir !
        print("Time for fit = ", self.time_fit)
        # Deplacement modele au bon endroit
        shutil.move("C:\\Users\\arnau\\Desktop\\quatrième_année\\Deep_Learning\\Projet_cifar-10\\model.png", main_directory+"\\logs_"+self.indiv_id+"\\model.png")

        return history
    
        
    
    def save_model(self, history, main_directory):
        # Sauvegarder le modele
        #plot_model(history, "model.png")
        #shutil.move("C:\\Users\\arnau\\Desktop\\quatrième_année\\Deep_Learning\\Projet_cifar-10\\model.png", "C:\\Users\\arnau\\Desktop\\quatrième_année\\Deep_Learning\\Projet_cifar-10\\"+self.log_folder+"\\logs"+str(self.indiv_id)+"\\model.png")
        
        # Afficher nos résultats dans un graphique matplotlib sauvegardé
        pyplot.gcf().subplots_adjust(hspace = 0.5)

        # Afficher la loss
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(history.history['loss'], color='blue', label='train')
        pyplot.plot(history.history['val_loss'], color='orange', label='test')
        # Afficher l'accuracy
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(history.history['accuracy'], color='blue', label='train')
        pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
        # Sauvegarde
        filename = main_directory+"\\logs_"+self.indiv_id+"\\"
        print("filename = ",filename)
        pyplot.savefig(filename + 'plot.png')
        pyplot.close()
       
        print("l2 = ", self.l2)
        
        print("LOSS : ", np.round(history.history['loss'],3))
        print("VAL_LOSS : ", np.round(history.history['val_loss'],3))
        print("ACCURACY : ", np.round(history.history['accuracy'],3))
        print("VAL_ACCURACY : ", np.round(history.history['val_accuracy'],3))
    
        # Créer un dataframe pandas que l'on va save en csv pour conserver les hyperparamètres du modèle
        df = pd.DataFrame({'indiv_id': [self.indiv_id],
                           'epochs': [self.epochs],
                           'nb_layers': [self.nb_layers],
                           'l1': [self.l1],
                           'l2': [self.l2],
                           'dropout': [self.dropout],
                           'filters_per_layers': [self.filters_per_layers],
                           'activation': [self.activation],
                           'kernel': [self.kernel],
                           'padding': [self.padding],
                           'max_pool': [self.max_pool],
                           'loss': np.round(history.history['loss'],3),
                           'val_loss': np.round(history.history['val_loss'],3),
                           'accuracy': np.round(history.history['accuracy'],3),
                           'val_accuracy': np.round(history.history['val_accuracy'],3),
                           'time_taken' : [self.time_fit]
                          })
        df.to_csv(path_or_buf=filename+"recap.csv",index=False)
    
    # Lance toutes les étapes
    def exec_indiv(self, main_directory):
        # Charger les données
        trainX, trainY, testX, testY = load_dataset()
        # Normaliser les données
        trainX, testX = prep_pixels(trainX, testX)
        # Créer et entrainer le modele
        model = self.create_and_train_model(trainX, trainY, testX, testY, main_directory)
        save = self.save_model(model, main_directory)
        # Evaluate model ou pas ? -> un peu long je trouve
        #_, acc = model.evaluate(testX, testY, verbose=0)
        #print('> %.3f' % (acc * 100.0))


# ### Classe Python qui va démarrer les tests des neural nets
# 

# In[75]:


# Classe générale qui va nous servir à effectuer des actions sur des listes d'individus
class MyTraining:
    # Prends un ID et une liste d'individus 
    def __init__(self, id_train, indiv_list):
        self.id_train = id_train
        self.indiv_list = indiv_list
    
    def train(self, main_directory):
        print("Start training\n")
        for idx in range(len(self.indiv_list)):
            print("indiv ", self.indiv_list[idx].indiv_id)
            self.indiv_list[idx].exec_indiv(main_directory)
            print("-----------------------------------------------------------------\n")
    
    def all_indiv(self):
        # Affiche les caractéristiques de l'ensemble des individus
        for indiv in self.indiv_list:
            print('\n'.join(indiv.__str__()))
            for tir in range(80): print('-', end='')
            print()


# ### Hyper paramètres
# 

# In[76]:


list_indiv_id = ['1', '2']
list_epochs = [1, 1]
list_nb_layers = [3, 0]
list_l1 = [0, 0]
list_l2 = [0, 0.001]
list_dropout = [0, 0]
list_filters_per_layers = [64, 32]
list_activation = ['relu', 'relu']
list_kernel = [(3,3), (3,3)]
list_padding = ['same', 'same']
list_max_pool = [0, 0]

main_directory =("C:\\Users\\arnau\\Desktop\\quatrième_année\\Deep_Learning\\Projet_cifar-10\\CONVNETS_"
                 +datetime.datetime.now().strftime("%Y%m%d-%H%M"))

# Afficher ici tous nos hyper paramètres dans un beau tableau ?


# ### Traitement général (train de l'ensemble des modèles)

# In[77]:


# Création des individus (des neurals nets, ici convnet)
list_indiv = []
for num in range(len(list_indiv_id)):
    list_indiv.append(IndividuConvnets(
        list_indiv_id[num],
          list_epochs[num],
          list_nb_layers[num],
          list_l1[num],
          list_l2[num],
          list_dropout[num],
          list_filters_per_layers[num],
          list_activation[num],
          list_kernel[num],
          list_padding[num],
          list_max_pool[num]
        )
    )
#ancienne méthode
#list_indiv.append(IndividuConvnets(1, 1, 3, 'false', 'false', 'false', 64, 'relu', (3,3), 'same', 0))
#list_indiv.append(IndividuConvnets(2, 1, 0, 'false', 'false', 'false', 32, 'relu', (3,3), 'same', 0))

# Chargement de la classe training, affichage des individus et train de tous les convnets
training_1 = MyTraining(1, list_indiv)
training_1.all_indiv()
training_1.train(main_directory)


# ### Partie tensorboard

# In[ ]:


# Procedure pour utiliser tensorboard ( j ai cherché longtemps pour trouver un truc aussi nul ;( )
#  1 load la première cell
#  2 utiliser la derniere cell avec --logdir (précisez bien votre répertoire, plus sur que ça
#    fonctionne avec une string "mon_path"
#  3 Vous NE POURREZ PLUS update tensorboard sur ce port et il y aura des bugs, pour éviter ça
#    quand vous voulez faire une update, fermez jupyter notebook (shutdown total) et réouvrez le 
#    OU, faites kernel->interrupt et changez de port + de folder de log
#PS : Oui, c'est de la merde


# In[ ]:


# Liste des ports utilisés par tensorboard, attention ça se remplit vite et il faut kill jupyter pour clean
from tensorboard import notebook
notebook.list()


# In[ ]:


# Tuez des processus si vous voulez, moi ça fonctionne po :(
#os.system("taskkill /im tensorboard.exe /f") #kill tous les processus qui utilisent tensorboard
#os.system('!kill 18776') #kill le processus X


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir "CONVNETS_20200113-2315/logs_20200113-231554/tensorboard_data" --port=6066')
# Code pour démarrer tensorboard dans le dossier souhaité [PRECISEZ BIEN LE DOSSIER ICI]


# In[ ]:


# Si vous avez la folie des grandeurs
notebook.display(port=6063, height=1000) 


# ### Fichier CSV Recap + Graphique

# In[ ]:


data_csv = pd.read_csv(main_directory+"\\logs_20200113-231554\\recap.csv")
data_csv.head()


# In[ ]:


image = pyplot.imread("C:\\Users\\arnau\\Desktop\\quatrième_année\\Deep_Learning\\Projet_cifar-10\\CONVNETS_20200113-1951\\logs1\\indiv1_plot.png")
pyplot.imshow(image)

