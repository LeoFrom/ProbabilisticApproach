#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import pickle #sauvegarder des objets dans des fichiers textes -> il faut un fichier binaire
data_sem = pickle.load(open("semantic_features.pkl", "rb"))
#-----------------------------------------------------------perceptron


corpus = open("Data_features.txt",encoding="UTF-8")

#creer train dev test

def create_train_test():
	train = []
	test = []
	data = []
	i=0
	for line in corpus:
		line = line.strip("\n")
		temp = line.split(",")
		label = temp[0]
		numeros = temp[1:]
		data.append((label, np.array(numeros, dtype = float)))

	random.shuffle(data)
	train = data[:310]
	test = data[310:]
	return train, test
	
train, test = create_train_test()


# fonction qui effectue la prediction/classification
def predict(v_obs, v_param): #prends vecteur complet et vparam
	#renvoie 1 si on predit que le binome est fige en faisant un produit scalaire entre un vecteur d'observation et le vecteur de paramètre 
	produit_scalaire = 0
	for feat in v_obs[1:]:
		for features in range(len(feat)):
		#on fait le produit sur les valeurs -1 ou 1 seulement, pas sur les strings representant les mots du binome
			try:
				produit_scalaire+= feat[features]*v_param[features]
			except:
				continue
	if (produit_scalaire>0):
		return 1
	return -1

# fonction learn
def learn(v_param):
	for vecteur in train:
		label = 0
		signe = predict(vecteur, v_param)
		
		#etiquette
		tuple_mots = (vecteur[0].split()[0], vecteur[0].split()[2])
		pourcentage_alpha = data_sem[tuple_mots]["alpha"] / (data_sem[tuple_mots]["alpha"] + data_sem[tuple_mots]["inverse"])
		pourcentage_inv = data_sem[tuple_mots]["inverse"] / (data_sem[tuple_mots]["inverse"] + data_sem[tuple_mots]["alpha"])

		#si ordre == True, alors il y a un ordre preferentiel
		ordre = pourcentage_alpha >= 0.7 or pourcentage_inv >= 0.7
		if (ordre):
			label = 1
		else:
			label = -1

		if (signe != label):
			#add_vect_param est une liste qui contient le résultat du produdit de l'étiquette et du vecteur d'observation 
			add_vect_param = []
			for feat in vecteur[1:]:
				for feat2feat in feat:
					add_vect_param.append(label * int(feat2feat))				
			v_param = [x + y for x, y in zip(v_param, add_vect_param)]
		#print(v_param)
	return v_param

# fonction evaluate

def evaluate(v_param):
	bonnes_reponses = 0
	for binome in test:
		tuple_mots = (binome[0].split()[0], binome[0].split()[2])
		pourcentage_alpha = data_sem[tuple_mots]["alpha"] / (data_sem[tuple_mots]["alpha"] + data_sem[tuple_mots]["inverse"])
		pourcentage_inv = data_sem[tuple_mots]["inverse"] / (data_sem[tuple_mots]["inverse"] + data_sem[tuple_mots]["alpha"])
		
		#si ordre == True, alors il y a un ordre preferentiel
		ordre = pourcentage_alpha >= 0.7 or pourcentage_inv >= 0.7
		
		pred = predict(binome, v_param)
		if (pred == 1 and ordre == True) or (pred == -1 and ordre == False):
			bonnes_reponses+=1

		rep =  bonnes_reponses/len(test) * 100
	return rep


def critere_erreur_naugmente_plus():
	v_param = [0,0,0,0,0,0,0,0,0,0,0,0]
	tour = 0
	score_max = 0
	while (int(score_max) != 100):
		for binome in train:
			pred = predict(binome, v_param)
			tuple_mots = (binome[0].split()[0], binome[0].split()[2])
			pourcentage_alpha = data_sem[tuple_mots]["alpha"] / (data_sem[tuple_mots]["alpha"] + data_sem[tuple_mots]["inverse"])
			pourcentage_inv = data_sem[tuple_mots]["inverse"] / (data_sem[tuple_mots]["inverse"] + data_sem[tuple_mots]["alpha"])
		
			#si ordre == True, alors il y a un ordre preferentiel
			ordre = pourcentage_alpha >= 0.7 or pourcentage_inv >= 0.7
			if (pred != 1 and ordre == True) or (pred != -1 and ordre == False):
				v_param = learn(v_param)
				
			tour+=1
			score_temp = evaluate(v_param)
			if score_max < score_temp:
				score_max = score_temp
				print(score_max, " tour:", tour, " v_param:", v_param)
		break
	
critere_erreur_naugmente_plus()