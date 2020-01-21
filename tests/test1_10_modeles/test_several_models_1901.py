#!/usr/bin/env python
# coding: utf-8

# ### Démarrage de tensorboard et imports principaux

# In[1]:


# Agrandir le notebook ?
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))

# Démarrage de tensorboard pour notebook
get_ipython().run_line_magic('load_ext', 'tensorboard')

import sys
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
import datetime
import os
import pandas as pd
from PIL import Image
import shutil  
from math import ceil, floor

from tensorflow.python.framework import ops #pour tenter de reset tensorboard, sans grand succès
ops.reset_default_graph()


# ### Fonctions pour préparer le dataset

# In[2]:


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

# In[10]:


# Classe pour les convnets
class IndividuConvnets:
    def __init__(self, indiv_id='1', epochs=10, batch_size=1, nb_layers=2, l1=0, l2=0, batch_norm=0, dropout=0, filters_per_layers=64, filters_double=6, MLP_end=0, activation='relu', kernel=(3,3), padding='same', max_or_avg_pool=0, learning_r=0.01, momentum=0.9, optimizer='SGD'):
        # Initialisation de nos variables
        self.time_fit = datetime.datetime.now()
        self.my_reguralizer = 'he_uniform'
        
        if nb_layers < 2:
            self.nb_layers = 2
        else:
            self.nb_layers = nb_layers
            
        self.loss = 0
        self.accuracy = 0
        self.indiv_id = indiv_id
        self.epochs = epochs
        self.batch_size = batch_size

        self.l1 = l1
        self.l2 = l2

        self.batch_norm = batch_norm
        self.dropout = dropout
        self.filters_per_layers = filters_per_layers
        
        # filters_double doit être supérieur ou égal à 2 OU égal à 0
        if filters_double < 2:
            self.filters_double = 0
        else : 
            self.filters_double = filters_double
        
        if MLP_end < 0:
            self.MLP_end = 0
        else:
            self.MLP_end = MLP_end
        
        self.activation = activation
        self.kernel = kernel
        self.padding = padding
        self.max_or_avg_pool = max_or_avg_pool
        self.learning_r = learning_r
        self.momentum = momentum
        self.optimizer = optimizer
    
    # ToString()
    def __str__(self):
        ma_liste = []
        ma_liste.append("indiv_id:{},\n ".format(self.indiv_id))
        ma_liste.append("epochs:{},\n ".format(self.epochs))
        ma_liste.append("batch_size:{},\n ".format(self.batch_size))
        ma_liste.append("nb_layers:{},\n ".format(self.nb_layers))
        ma_liste.append("l1:{},\n ".format(self.l1))
        ma_liste.append("l2:{},\n ".format(self.l2))
        ma_liste.append("batch_norm:{},\n ".format(self.batch_norm))
        ma_liste.append("dropout:{},\n ".format(self.dropout))
        ma_liste.append("filters_per_layers:{},\n ".format(self.filters_per_layers))
        ma_liste.append("filters_double:{},\n ".format(self.filters_double))
        ma_liste.append("MLP_end:{},\n ".format(self.MLP_end))
        ma_liste.append("activation:{},\n ".format(self.activation))
        ma_liste.append("kernel:\n ")
        ma_liste.append("{},\n ".format(self.kernel))
        ma_liste.append("padding:{},\n ".format(self.padding))
        ma_liste.append("max_or_avg_pool:{}\n".format(self.max_or_avg_pool))
        ma_liste.append("learning_r:{}\n".format(self.learning_r))
        ma_liste.append("momentum:{}\n".format(self.momentum))
        ma_liste.append("optimizer:{}\n".format(self.optimizer))
            
        return ma_liste
    
    # (Modele 2 conv + norm ? + pool) * X -> MLP -> softmax sortie 10 -> MODELE BLOC 2
    # D'autres modeles seront crees par la suite
    def create_and_train_model(self, trainX, trainY, testX, testY, main_directory):
        start = datetime.datetime.now()
        
        # Update indiv_id pour avoir un vrai ID unique
        self.indiv_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Choix d'un emplacement pour les logs
        log_dir=main_directory+"\\logs_"+self.indiv_id+"\\tensorboard_data\\"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        print("log dir = ",log_dir)
        
        # l1 et l2
        if self.l1 > 0 and self.l2 > 0:
            self.my_regularizer = regularizers.l1_l2(l1=self.l1 / self.nb_layers, l2=self.l2 / self.nb_layers)
        if self.l1 > 0:
            self.my_regularizer = regularizers.l1(self.l1 / self.nb_layers)
        elif self.l2 > 0:
            self.my_regularizer = regularizers.l2(self.l2 / self.nb_layers)
        else:
            self.my_reguralizer = 'he_uniform'
            
        # Definir notre modèle basique
        model = Sequential()

        # Faire toutes les convs nécessaires (conv * 2 + max pool)
        double_count = 0 # Var pour doubler les filtres
        
        for i in range(0, self.nb_layers):
            # mieux gérer les ids
            if i % 2 != 0:
                continue
            # Traitement pour doubler les filtres
            if double_count >= self.filters_double and self.filters_double != 0:
                self.filters_per_layers = self.filters_per_layers * 2
                print("filters = ", self.filters_per_layers)
                double_count = 0
            
            # Choix du bloc (2 conv pool ou 1 conv pool si nb impair)
            if self.nb_layers - i != 1:
                print("i = ",i, " 2 pools")
                # 2 conv + pool
                model.add(Conv2D(self.filters_per_layers, self.kernel, activation=self.activation, kernel_initializer=self.my_reguralizer, padding=self.padding,input_shape=(32, 32, 3), name='conv_'+str(self.filters_per_layers)+'_'+str(i)))
                model.add(Conv2D(self.filters_per_layers, self.kernel, activation=self.activation, kernel_initializer=self.my_reguralizer, padding=self.padding,input_shape=(32, 32, 3), name='conv_'+str(self.filters_per_layers)+'_'+str(i+1)))
                
                if self.batch_norm == 1:
                    model.add(BatchNormalization( name='batchnorm_'+str(i/2)))
                
                # Max ou Avg pooling
                if self.max_or_avg_pool == 'max':
                    model.add(MaxPooling2D((2, 2), name='max_pool_'+str(i/2)))
                else:
                    model.add(AveragePooling2D((2, 2), name='avg_pool_'+str(i/2)))
                
            else:
                print("i = ",i, " 1 pool")
                # 1 conv + pool si nombre impair de couches (nb_layers)
                model.add(Conv2D(self.filters_per_layers, self.kernel, activation=self.activation, kernel_initializer=self.my_reguralizer, padding=self.padding,input_shape=(32, 32, 3), name='conv_'+str(self.filters_per_layers)+'_'+str(i)))
                if self.batch_norm == 1:
                    model.add(BatchNormalization(name='batchnorm_'+str(ceil(i/2))))
                
                # Max or Avg pooling
                if self.max_or_avg_pool == 'max':
                    model.add(MaxPooling2D((2, 2), name='max_pool_'+str(ceil(i/2))))
                else:
                    model.add(AveragePooling2D((2, 2), name='avg_pool_'+str(ceil(i/2))))
                    
            double_count = double_count + 2
        
        
        # Fin des convs -> neural network classique
        model.add(Flatten(name='Flatten'))
        
        #tTrain dans un MLP avant la fin si on le souhaite
        if self.MLP_end > 0:
            model.add(Dense(128, activation='relu', kernel_initializer=self.my_reguralizer, name='MLP_'+str(self.MLP_end)))

            #mettre dropout sur les Dense, pas opti sur les convnets (mais on peut le faire pour le démontrer ??)
            if self.dropout > 0:
                model.add(Dropout(self.dropout))
        
        #notre output
        model.add(Dense(10, activation='softmax', name='output')) 

        # Compiler le modele
        if self.optimizer == 'SGD':
            print("SGD, learning_r = ", self.learning_r, " momentum = ", self.momentum, "\n")
            opt = SGD(lr=self.learning_r, momentum=self.momentun)
        else:
            print("Adam learning_r = ", self.learning_r, " momentum = ", self.momentum, "\n")
            opt = Adam(lr=self.learning_r, beta_1=self.momentum) # beta_1 => notation pour momentum Adam
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Entrainer le modele
        history = model.fit(trainX, trainY, epochs=self.epochs, batch_size=self.batch_size, validation_data=(testX, testY), verbose=0, callbacks=[tensorboard_callback])
        
        # Garder une trace du temps nécessaire pour fit (peut être pas la meilleure méthode)
        end = datetime.datetime.now()
        self.time_fit = end - start
        print("\nTime for fit = ", round(self.time_fit.total_seconds(),2)) # Round avec total_seconds()

        return history, model
    
    
    def save_model(self, history, model, main_directory):
        
        # Sauvegarde du modèle
        plot_model(model, "model.png")
        
         # Deplacement modele au bon endroit
        shutil.move("C:\\Users\\arnau\\Desktop\\quatrième_année\\Deep_Learning\\Projet_cifar-10\\model.png", main_directory+"\\logs_"+self.indiv_id+"\\model.png")
        
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
        pyplot.savefig(filename + 'plot.png')
        pyplot.close()
       
        
        print("LOSS : ", round(history.history['loss'][0].item(), 3))
        print("VAL_LOSS : ", round(history.history['val_loss'][0].item(), 3))
        print("ACCURACY : ", round(history.history['accuracy'][0].item(), 3))
        print("VAL_ACCURACY : ", round(history.history['val_accuracy'][0].item(), 3))
        
        # attributs pour créer les csv indivudels et le csv global
        self.loss = round(history.history['loss'][0].item(), 3)
        self.val_loss = round(history.history['val_loss'][0].item(), 3)
        self.accuracy = round(history.history['accuracy'][0].item(), 3)
        self.val_accuracy = round(history.history['val_accuracy'][0].item(), 3)
        self.time_taken = round(self.time_fit.total_seconds(),2)
        
        # Créer un dataframe pandas (avec hyperparams) et le sauvegarder en CSV
        df = pd.DataFrame({'indiv_id': [self.indiv_id],
                           'epochs': [self.epochs],
                           'batch_size': [self.batch_size],
                           'nb_layers': [self.nb_layers],
                           'l1': [self.l1],
                           'l2': [self.l2],
                           'batch_norm': [self.batch_norm],
                           'dropout': [self.dropout],
                           'filters_per_layers': [self.filters_per_layers],
                           'filters_double': [self.filters_double],
                           'MLP_end': [self.MLP_end],
                           'activation': [self.activation],
                           'kernel': [self.kernel],
                           'padding': [self.padding],
                           'max_or_avg_pool': [self.max_or_avg_pool],
                           'loss': [self.loss],
                           'val_loss': [self.val_loss],
                           'accuracy': [self.accuracy],
                           'val_accuracy': [self.val_accuracy],
                           'time_taken' : [self.time_taken],
                           'learning_r' : [self.learning_r],
                           'momentum' : [self.momentum],
                           'optimizer' : [self.optimizer]
                          })
        
        df.to_csv(path_or_buf=filename+"recap.csv",index=False)
    
    # Lance toutes les étapes
    def exec_indiv(self, main_directory):
        
        # Charger les données
        trainX, trainY, testX, testY = load_dataset()
        
        # Normaliser les données
        trainX, testX = prep_pixels(trainX, testX)
        
        # Créer et entrainer le modele
        history, model = self.create_and_train_model(trainX, trainY, testX, testY, main_directory)
        
        # Sauvegarder le modèle
        save = self.save_model(history, model, main_directory)


# ### Classe Python qui va démarrer les tests des neural nets
# 

# In[4]:


# Classe générale qui va nous servir à effectuer des actions sur des individus
class MyTraining:
    # Prends un ID et une liste d'individus 
    def __init__(self, id_train, indiv_list):
        
        self.id_train = id_train
        self.indiv_list = indiv_list
    
    def train(self, main_directory):
        
        print("Start training\n")
        
        for indiv in self.indiv_list:
            print("indiv ", indiv.indiv_id, "\n")
            indiv.exec_indiv(main_directory)
            print("-----------------------------------------------------------------\n")
        
        # Fusion des csv 
        merge_csv = pd.DataFrame(columns=['indiv_id', 'epochs', 'nb_layers', 'l1', 'l2', 'batch_norm', 'dropout',
                                          'filters_per_layers', 'filters_double', 'MLP_end', 'activation', 'kernel',
                                          'padding','max_or_avg_pool','loss', 'val_loss', 'accuracy', 'val_accuracy',
                                          'time_taken','learning_r', 'momentum', 'optimizer'])
        
        for indiv in self.indiv_list:
            merge_csv = merge_csv.append({'indiv_id': indiv.indiv_id, 'epochs': indiv.epochs, 'batch_size': indiv.batch_size,
                              'nb_layers' : indiv.nb_layers,'l1' : indiv.l1, 'l2' : indiv.l2, 'batch_norm': indiv.batch_norm,
                              'dropout' : indiv.dropout,'filters_per_layers' : indiv.filters_per_layers,
                              'filters_double' : indiv.filters_double,'MLP_end' : indiv.MLP_end,'activation' : indiv.activation,
                              'kernel' : indiv.kernel,'padding' : indiv.padding, 'max_or_avg_pool' : indiv.max_or_avg_pool,
                              'loss' : indiv.loss,'val_loss' : indiv.val_loss, 'accuracy' : indiv.accuracy, 
                              'val_accuracy' : indiv.val_accuracy,'time_taken' : indiv.time_taken, 'learning_r' : indiv.learning_r,
                              'momentum': indiv.momentum, 'optimizer' : indiv.optimizer},ignore_index=True)
        
        # sauvegarde
        merge_csv.to_csv(main_directory+"\\combined_recap.csv", index=False)
            
    
    def all_indiv(self):
        
        # Affiche les caractéristiques de l'ensemble des individus
        for indiv in self.indiv_list:
            print('\n'.join(indiv.__str__()))
            for tir in range(80): print('-', end='')
            print()


# ### Hyper paramètres
# 

# In[5]:


# Parametres de verification : 

#list_indiv_id = ['1', '2']
#list_epochs = [1, 1]
#list_batch_size = [100, 64]
#list_nb_layers = [6, 2]
#list_l1 = [0.01, 0]
#list_l2 = [0.01, 0.001]
#list_batch_norm = [0, 1]
#list_dropout = [0, 0.2]
#list_filters_per_layers = [64, 32]
#list_filters_double = [2, 0]
#list_MLP_end = [120, 0]
#list_activation = ['relu', 'relu']
#list_kernel = [(3,3), (3,3)]
#list_padding = ['same', 'same']
#list_max_or_avg_pool = ['max', 'avg']
#list_learning_r = [0.1, 0.01]
#list_momentum = [0.9, 0.85]
#list_optimizer = ['SGD', 'Adam']

#main_directory =("C:\\Users\\arnau\\Desktop\\quatrième_année\\Deep_Learning\\Projet_cifar-10\\CONVNETS_"
#                 +datetime.datetime.now().strftime("%Y%m%d-%H%M"))


# In[11]:


#Premier Test :
# CONSTANTES : nb_layers = 8, batch_size = 50, epochs 100, lr = 0.01, momentum = 0.9, optimizer Adam, 
#padding = same, maxpool, relu, kernel = (3,3)
# 
# * 3 convnets sans regularization et MLP à 128
#   - 2 filters double(2) avec filters (16, 32) 
#   - 1 sans filters double avec filters (32)

# 2 convnets sans regu ou on test le MLP_end et filters 64
#  * 1 sans double, filter 64 avec MLP_end(128)
#  * 1 sans double, filter 64 avec MLP_end(0)

# 5 convnets avec regu (+ MLP à 128) et filters 32 sans double
#  * 1 convnet avec l1 à 0.01
#  * 1 convnet avec l1 à 0.01 et batchnorm
#  * 1 convnet avec l2 à 0.01 et batchnorm
#  * 1 convnet avec L1 et L2 à 0.01 et batchnorm
#  * 1 convnet avec L1 et L2 à 0.01 + batchnorm + dropout à 0.2

# LEXIQUE PARAM : 
# * filters_double permet de savoir toutes les combien de couche on double les filtres, si 0 on double pas

list_indiv_id = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
list_epochs = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
list_batch_size = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
list_nb_layers = [8,8,8,8,8,8,8,8,8,8]
list_l1 = [0, 0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01, 0.01]
list_l2 = [0, 0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01]
list_batch_norm = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
list_dropout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2]
list_filters_per_layers = [16, 32, 32, 64, 64, 32, 32, 32, 32, 32]
list_filters_double = [2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
list_MLP_end = [128, 128, 128, 128, 0, 128, 128, 128, 128, 128]
list_activation = ['relu','relu','relu','relu','relu','relu','relu','relu','relu','relu']
list_kernel = [(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3)]
list_padding = ['same','same','same','same','same','same','same','same','same','same']
list_max_or_avg_pool = ['max','max','max','max','max','max','max','max','max','max']
list_learning_r = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
list_momentum = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]
list_optimizer = ['Adam','Adam','Adam','Adam','Adam','Adam','Adam','Adam','Adam','Adam']

main_directory =("C:\\Users\\arnau\\Desktop\\quatrième_année\\Deep_Learning\\Projet_cifar-10\\CONVNETS_"
                 +datetime.datetime.now().strftime("%Y%m%d-%H%M"))


# ### Traitement général (train de l'ensemble des modèles)

# In[12]:


# Création des individus (des neurals nets, ici convnet)
list_indiv = []
for num in range(len(list_indiv_id)):
    list_indiv.append(IndividuConvnets(
        list_indiv_id[num],
          list_epochs[num],
          list_batch_size[num],
          list_nb_layers[num],
          list_l1[num],
          list_l2[num],
          list_batch_norm[num],
          list_dropout[num],
          list_filters_per_layers[num],
          list_filters_double[num],
          list_MLP_end[num],
          list_activation[num],
          list_kernel[num],
          list_padding[num],
          list_max_or_avg_pool[num],
          list_learning_r[num],
          list_momentum[num],
          list_optimizer[num]
        )
    )

# Chargement de la classe training, affichage des individus et train de tous les convnets
training_1 = MyTraining(1, list_indiv)
training_1.all_indiv()
training_1.train(main_directory)


# ### Partie tensorboard

# In[ ]:


# Procedure pour utiliser tensorboard
#  1 load la première cell
#  2 utiliser la derniere cell avec --logdir (précisez bien votre répertoire, plus sur que ça
#    fonctionne avec une string "mon_path"
#  3 Vous NE POURREZ PLUS update tensorboard sur ce port et il y aura des bugs, pour éviter ça
#    quand vous voulez faire une update, fermez jupyter notebook (shutdown total) et réouvrez le 
#    OU, faites kernel->interrupt et changez de port + de folder de log

#si vous voulez tenter de tuer des process
#os.system("taskkill /im tensorboard.exe /f") #kill tous les processus qui utilisent tensorboard
#os.system('!kill 18776') #kill le processus X


# In[ ]:


# Liste des ports utilisés par tensorboard, attention ça se remplit vite et il faut kill jupyter pour clean
from tensorboard import notebook
notebook.list()


# In[ ]:


# Code pour démarrer tensorboard dans le dossier souhaité [PRECISEZ BIEN LE DOSSIER ICI]
get_ipython().run_line_magic('tensorboard', '--logdir "CONVNETS_20200116-2354\\logs_20200116-235500\\tensorboard_data" --port=6062')


# In[ ]:


# Si vous avez la folie des grandeurs
notebook.display(port=6066, height=1000) 


# In[ ]:


### Fichier CSV combined_recap + Graphique


# In[ ]:


# Commandes pandas utiles

#data_csv = pd.read_csv(main_directory+"\\combined_recap.csv")

#meilleure accuracy, moins pire loss par ex
#data_csv.sort_values(["elapsed"], axis=0, 
                 #ascending=[False], inplace=True) 

# Afficher uniquement certaines colonnes
#dataX = data_csv.filter(items=['elapsed', 'label'])

#récupérer uniquement où la loss est < à X et ou kernel = (3,3) par exemple
#dataX = data_csv.loc[(data_csv['elapsed'] > 700) & (data_csv['threadName'].str.contains('Thread Group 1-2'))]
#dataX

#pd.set_option('display.max_rows', data3.shape[0]+1) #nombre de row max à afficher
#data_csv = pd.read_csv(main_directory+"\\logs_20200116-204456\\recap.csv")
#data_csv.head()


# In[ ]:


image = pyplot.imread("C:\\Users\\arnau\\Desktop\\quatrième_année\\Deep_Learning\\Projet_cifar-10\\CONVNETS_20200113-1951\\logs1\\indiv1_plot.png")
pyplot.imshow(image)

