import sys
import argparse
import bz2
import os
import pprint as pp
import csv
from collections import defaultdict, Counter
from numpy import mean, std
import collections
import numpy as np
import pandas as pd
import seaborn as sns #mère de matplotlib
import pickle
import json
import matplotlib.pyplot as plt #représentation graphique
import matplotlib #code de bas niveau: on doit bcp coder en dur
import pickle #sauvegarder des objets dans des fichiers textes -> il faut un fichier binaire
data_sem = pickle.load(open("semantic_features.pkl", "rb"))
data_phon = json.loads(open("annotations_phon.json".format()).read())

# chaîne de caractères qui sera affichée si l'option -h est précisée, ou si les options passées ne sont pas celles déclarées
usage_str = sys.argv[0]
# --------------------------------
# gestion des options et des arguments
argparser = argparse.ArgumentParser(usage = usage_str)

# déclaration d'un argument
argparser.add_argument('corpus_AA', help ='répertoire de corpus wiki compressés', default = None)
argparser.add_argument('subtitles', help ='répertoire de corpus de sous-titres compressés', default = None)

# lecture des arguments et des options passés en ligne de commande
args = argparser.parse_args()

try:
	allDocs = os.listdir(args.corpus_AA) #(alterner AA ou Test)
except IOError:
    print ("Impossible d'ouvrir " + allDocs)
    exit()

try:
	allSubtitles = args.subtitles
except IOError:
	print("Impossible d'ouvrir" + allSubtitles)
	exit()


def is_number(string):
	"""Fonction qui renvoie si une string est un nombre

    :param: une string
    :type : string
    :return: True si c'est un chiffre false sinon
    :rtype: boolean
    """
	try:
		int(string)
		float(string)
		return True
	except ValueError:
		return False

def lecture_wiki(corpus):
	"""Fonction qui fait la lecture d'un dossier contenant des fichiers conll

    :param: Corpus étant un dossier
    :type : File.conll
    :return: Retourne 2 dictionnaires contenant les binomes, les catégories avec leurs occurences respectives
    :rtype: defaultdict
    """
	AllDictionnaryWord = defaultdict(int)
	AllDictionnaryCat = defaultdict(int)

	for element in corpus: #on lit chaque element dans notre dossier
		f = open(args.corpus_AA + element,encoding="UTF-8")
		while(element != None):
			flux = []
			mot_decode = f.readline().strip().split("\t")
			while mot_decode != [""]:
				flux.append(mot_decode)
				mot_decode = f.readline().strip().split("\t")
			
			if(flux == []):
				break;

			for i in range(len(flux)-4):
				#besoin du decode pour les fichiers zippé du corpus AA
				mot_actu = flux[i] #on enlève les saut de ligne et séparation par tab
				mot_actu_plus1 = flux[i+1]
				mot_actu_plus2 = flux[i+2]
				#pour une binomial de taille 3 au total
				#actu +1 +2
				#ou +1 +2//et/ou +3 (fin)
				#ou +2 +3//et/ou +4 (fin)

				mot_actu_plus3 = flux[i+3]
				mot_actu_plus4 = flux[i+4]
				#pour une binomial de taille 5 au total
				#actu +1 +2//et/ou +3 +4

				#supprimer les dates 
			
				if(mot_actu != [""] and mot_actu_plus1 != [""] and mot_actu_plus2 != [""] and mot_actu_plus3 != [""] and mot_actu_plus4 != [""] and is_number(mot_actu[1])==False and is_number(mot_actu_plus4[1])==False and len(mot_actu_plus1[1])>=2 and len(mot_actu_plus4[1])>=2):
						
					if(mot_actu_plus1[1] == "et"  or mot_actu_plus1[1] == "ou"): #on regarde les binomiaux a 3 mots
						#XX[1] equivaut au mot et XX[4] equivaut à la cat
						if(mot_actu[4] == "ADJ" and mot_actu_plus2[4] == "ADJ"):
							#on ajoute dans dicocat et dico word les binomiaux que on a trouvé
							AllDictionnaryCat[mot_actu[4],mot_actu_plus1[4],mot_actu_plus2[4]] +=1
							AllDictionnaryWord[mot_actu[1]+"/"+mot_actu[4], mot_actu_plus1[1]+"/"+mot_actu_plus1[4], mot_actu_plus2[1]+"/"+mot_actu_plus2[4]] +=1

						if(mot_actu[4] == "NC" and mot_actu_plus2[4] == "NC"):
							AllDictionnaryCat[mot_actu[4],mot_actu_plus1[4],mot_actu_plus2[4]] +=1
							AllDictionnaryWord[mot_actu[1]+"/"+mot_actu[4],mot_actu_plus1[1]+"/"+mot_actu_plus1[4],mot_actu_plus2[1]+"/"+mot_actu_plus2[4]] +=1

						if(mot_actu[4] == "VPP" and mot_actu_plus2[4] == "VPP"):
							AllDictionnaryCat[mot_actu[4],mot_actu_plus1[4],mot_actu_plus2[4]] +=1
							AllDictionnaryWord[mot_actu[1]+"/"+mot_actu[4],mot_actu_plus1[1]+"/"+mot_actu_plus1[4],mot_actu_plus2[1]+"/"+mot_actu_plus2[4]] +=1
							
						if(mot_actu[4] == "V" and mot_actu_plus2[4] == "V"):
							AllDictionnaryCat[mot_actu[4],mot_actu_plus1[4],mot_actu_plus2[4]] +=1
							AllDictionnaryWord[mot_actu[1]+"/"+mot_actu[4],mot_actu_plus1[1]+"/"+mot_actu_plus1[4],mot_actu_plus2[1]+"/"+mot_actu_plus2[4]] +=1

					if(i == len(flux)-4): #si on est à la fin on verifie les derniers binomiaux a 3 mots (comme on s'arrête à flux-4)
						if(mot_actu_plus2[1] == "et"  or mot_actu_plus2[1] == "ou"):
							if(mot_actu_plus1[4] == "ADJ" and mot_actu_plus3[4] == "ADJ"):
							#on ajoute dans dicocat et dico word les binomiaux que on a trouvé
								AllDictionnaryCat[mot_actu_plus1[4],mot_actu_plus2[4],mot_actu_plus3[4]] +=1
								AllDictionnaryWord[mot_actu_plus1[1]+"/"+mot_actu_plus1[4],mot_actu_plus2[1]+"/"+mot_actu_plus2[4],mot_actu_plus3[1]+"/"+mot_actu_plus3[4]] +=1

							if(mot_actu_plus1[4] == "NC" and mot_actu_plus3[4] == "NC"):
								AllDictionnaryCat[mot_actu_plus1[4],mot_actu_plus2[4],mot_actu_plus3[4]] +=1
								AllDictionnaryWord[mot_actu_plus1[1]+"/"+mot_actu_plus1[4],mot_actu_plus2[1]+"/"+mot_actu_plus2[4],mot_actu_plus3[1]+"/"+mot_actu_plus3[4]] +=1

							if(mot_actu_plus1[4] == "VPP" and mot_actu_plus3[4] == "VPP"):
								AllDictionnaryCat[mot_actu_plus1[4],mot_actu_plus2[4],mot_actu_plus3[4]] +=1
								AllDictionnaryWord[mot_actu_plus1[1]+"/"+mot_actu_plus1[4],mot_actu_plus2[1]+"/"+mot_actu_plus2[4],mot_actu_plus3[1]+"/"+mot_actu_plus3[4]] +=1
							
							if(mot_actu_plus1[4] == "V" and mot_actu_plus3[4] == "V"):
								AllDictionnaryCat[mot_actu_plus1[4],mot_actu_plus2[4],mot_actu_plus3[4]] +=1
								AllDictionnaryWord[mot_actu_plus1[1]+"/"+mot_actu_plus1[4],mot_actu_plus2[1]+"/"+mot_actu_plus2[4],mot_actu_plus3[1]+"/"+mot_actu_plus3[4]] +=1

						if(mot_actu_plus3[1] == "et"  or mot_actu_plus3[1] == "ou"):
							if(mot_actu_plus2[4] == "ADJ" and mot_actu_plus4[4] == "ADJ"):
							#on ajoute dans dicocat et dico word les binomiaux que on a trouvé
								AllDictionnaryCat[mot_actu_plus2[4],mot_actu_plus3[4],mot_actu_plus4[4]] +=1
								AllDictionnaryWord[mot_actu_plus2[1]+"/"+mot_actu_plus2[4],mot_actu_plus3[1]+"/"+mot_actu_plus3[4],mot_actu_plus4[1]+"/"+mot_actu_plus4[4]] +=1

							if(mot_actu_plus2[4] == "NC" and mot_actu_plus4[4] == "NC"):
								AllDictionnaryCat[mot_actu_plus2[4],mot_actu_plus3[4],mot_actu_plus4[4]] +=1
								AllDictionnaryWord[mot_actu_plus2[1]+"/"+mot_actu_plus2[4],mot_actu_plus3[1]+"/"+mot_actu_plus3[4],mot_actu_plus4[1]+"/"+mot_actu_plus4[4]] +=1

							if(mot_actu_plus2[4] == "VPP" and mot_actu_plus4[4] == "VPP"):
								AllDictionnaryCat[mot_actu_plus2[4],mot_actu_plus3[4],mot_actu_plus4[4]] +=1
								AllDictionnaryWord[mot_actu_plus2[1]+"/"+mot_actu_plus2[4],mot_actu_plus3[1]+"/"+mot_actu_plus3[4],mot_actu_plus4[1]+"/"+mot_actu_plus4[4]] +=1
							
							if(mot_actu_plus2[4] == "V" and mot_actu_plus4[4] == "V"):
								AllDictionnaryCat[mot_actu_plus2[4],mot_actu_plus3[4],mot_actu_plus4[4]] +=1
								AllDictionnaryWord[mot_actu_plus2[1]+"/"+mot_actu_plus2[4],mot_actu_plus3[1]+"/"+mot_actu_plus3[4],mot_actu_plus4[1]+"/"+mot_actu_plus4[4]] +=1

					if(mot_actu_plus2[1] == "et" or mot_actu_plus2[1] == "ou"): #on regarde les binomiaux à 5 mots
						if(mot_actu[4] == "DET" and mot_actu_plus1[4] == "NC" and mot_actu_plus3[4] == "DET" and mot_actu_plus4[4] == "NC"):
							AllDictionnaryCat[mot_actu[4],mot_actu_plus1[4],mot_actu_plus2[4],mot_actu_plus3[4],mot_actu_plus4[4]]+=1
							AllDictionnaryWord[mot_actu[1]+"/"+mot_actu_plus1[4],mot_actu_plus1[1]+"/"+mot_actu_plus1[4],mot_actu_plus2[1]+"/"+mot_actu_plus2[4],mot_actu_plus3[1]+"/"+mot_actu_plus3[4],mot_actu_plus4[1]+"/"+mot_actu_plus4[4]]+=1

						if(mot_actu[4] == "P" and mot_actu_plus1[4] == "NC" and mot_actu_plus3[4] == "P" and mot_actu_plus4[4] == "NC"):
							AllDictionnaryCat[mot_actu[4],mot_actu_plus1[4],mot_actu_plus2[4],mot_actu_plus3[4],mot_actu_plus4[4]]+=1
							AllDictionnaryWord[mot_actu[1]+"/"+mot_actu[4],mot_actu_plus1[1]+"/"+mot_actu_plus1[4],mot_actu_plus2[1]+"/"+mot_actu_plus2[4],mot_actu_plus3[1]+"/"+mot_actu_plus3[4],mot_actu_plus4[1]+"/"+mot_actu_plus4[4]]+=1
	
	titre = 'Dico_AA_words.csv'
	w1 = csv.writer(open(titre,'w',encoding="UTF-8"))
	for clef_mot, occu_mot in AllDictionnaryWord.items():
		if(occu_mot != 3): #on va prendre 3 comme référence de binomes "non nécessaire"
			w1.writerow([clef_mot,occu_mot])

	titre = 'Dico_AA_cats.csv'
	w2 = csv.writer(open(titre,'w',encoding="UTF-8"))
	for clef_cat, occu_cat in AllDictionnaryCat.items():
			w2.writerow([clef_cat,occu_cat])


	return (AllDictionnaryWord, AllDictionnaryCat)

def lecture_sub(corpus):
	"""Fonction qui fait la lecture d'un fichier annoté de sous-titres

    :param: Corpus étant un fichier de sous-titres
    :type : File.pos
    :return: Retourne 2 dictionnaires contenant les binomes, les catégories avec leurs occurences respectives
    :rtype: defaultdict
    """
	AllDictionnaryWord = defaultdict(int)
	AllDictionnaryCat = defaultdict(int)

	for element in corpus: #on lit chaque element dans notre dossier
		f = open(corpus, encoding="UTF-8") #dossier subtitles
		while(element != None):
			flux = []
			mot_decode = f.readline().strip().split("\t")
			while mot_decode != [""]:
				flux.append(mot_decode)
				mot_decode = f.readline().strip().split("\t")
			
			if(flux == []):
				break;

			for i in range(len(flux)-4):

				#pas besoin de decode ici
				mot_actu = flux[i] #on enlève les saut de ligne et séparation par tab
				mot_actu_plus1 = flux[i+1]
				mot_actu_plus2 = flux[i+2]
				#pour une binomial de taille 3 au total
				#actu +1 +2
				#ou +1 +2//et/ou +3 (fin)
				#ou +2 +3//et/ou +4 (fin)

				mot_actu_plus3 = flux[i+3]
				mot_actu_plus4 = flux[i+4]
				#pour une binomial de taille 5 au total
				#actu +1 +2//et/ou +3 +4

				#supprimer les dates 
			
				if(mot_actu != [""] and mot_actu_plus1 != [""] and mot_actu_plus2 != [""] and mot_actu_plus3 != [""] and mot_actu_plus4 != [""] and is_number(mot_actu[1])==False and is_number(mot_actu_plus4[1])==False and len(mot_actu_plus1[1])>=2 and len(mot_actu_plus4[1])>=2):
						
					if(mot_actu_plus1[1] == "et"  or mot_actu_plus1[1] == "ou"): #on regarde les binomiaux a 3 mots
						#XX[1] equivaut au mot et XX[4] equivaut à la cat
						if(mot_actu[3] == "ADJ" and mot_actu_plus2[3] == "ADJ"):
							#on ajoute dans dicocat et dico word les binomiaux que on a trouvé
							AllDictionnaryCat[mot_actu[3],mot_actu_plus1[3],mot_actu_plus2[3]] +=1
							AllDictionnaryWord[mot_actu[1]+"/"+mot_actu[3], mot_actu_plus1[1]+"/"+mot_actu_plus1[3], mot_actu_plus2[1]+"/"+mot_actu_plus2[3]] +=1

						if(mot_actu[3] == "NC" and mot_actu_plus2[3] == "NC"):
							AllDictionnaryCat[mot_actu[3],mot_actu_plus1[3],mot_actu_plus2[3]] +=1
							AllDictionnaryWord[mot_actu[1]+"/"+mot_actu[3],mot_actu_plus1[1]+"/"+mot_actu_plus1[3],mot_actu_plus2[1]+"/"+mot_actu_plus2[3]] +=1

						if(mot_actu[3] == "VPP" and mot_actu_plus2[3] == "VPP"):
							AllDictionnaryCat[mot_actu[3],mot_actu_plus1[3],mot_actu_plus2[3]] +=1
							AllDictionnaryWord[mot_actu[1]+"/"+mot_actu[3],mot_actu_plus1[1]+"/"+mot_actu_plus1[3],mot_actu_plus2[1]+"/"+mot_actu_plus2[3]] +=1
							
						if(mot_actu[3] == "V" and mot_actu_plus2[3] == "V"):
							AllDictionnaryCat[mot_actu[3],mot_actu_plus1[3],mot_actu_plus2[3]] +=1
							AllDictionnaryWord[mot_actu[1]+"/"+mot_actu[3],mot_actu_plus1[1]+"/"+mot_actu_plus1[3],mot_actu_plus2[1]+"/"+mot_actu_plus2[3]] +=1

					if(i == len(flux)-4): #si on est à la fin on verifie les derniers binomiaux a 3 mots (comme on s'arrête à flux-4)
						if(mot_actu_plus2[1] == "et"  or mot_actu_plus2[1] == "ou"):
							if(mot_actu_plus1[3] == "ADJ" and mot_actu_plus3[3] == "ADJ"):
							#on ajoute dans dicocat et dico word les binomiaux que on a trouvé
								AllDictionnaryCat[mot_actu_plus1[3],mot_actu_plus2[3],mot_actu_plus3[3]] +=1
								AllDictionnaryWord[mot_actu_plus1[1]+"/"+mot_actu_plus1[3],mot_actu_plus2[1]+"/"+mot_actu_plus2[3],mot_actu_plus3[1]+"/"+mot_actu_plus3[3]] +=1

							if(mot_actu_plus1[3] == "NC" and mot_actu_plus3[3] == "NC"):
								AllDictionnaryCat[mot_actu_plus1[3],mot_actu_plus2[3],mot_actu_plus3[3]] +=1
								AllDictionnaryWord[mot_actu_plus1[1]+"/"+mot_actu_plus1[3],mot_actu_plus2[1]+"/"+mot_actu_plus2[3],mot_actu_plus3[1]+"/"+mot_actu_plus3[3]] +=1

							if(mot_actu_plus1[3] == "VPP" and mot_actu_plus3[3] == "VPP"):
								AllDictionnaryCat[mot_actu_plus1[3],mot_actu_plus2[3],mot_actu_plus3[3]] +=1
								AllDictionnaryWord[mot_actu_plus1[1]+"/"+mot_actu_plus1[3],mot_actu_plus2[1]+"/"+mot_actu_plus2[3],mot_actu_plus3[1]+"/"+mot_actu_plus3[3]] +=1
							
							if(mot_actu_plus1[3] == "V" and mot_actu_plus3[3] == "V"):
								AllDictionnaryCat[mot_actu_plus1[3],mot_actu_plus2[3],mot_actu_plus3[3]] +=1
								AllDictionnaryWord[mot_actu_plus1[1]+"/"+mot_actu_plus1[3],mot_actu_plus2[1]+"/"+mot_actu_plus2[3],mot_actu_plus3[1]+"/"+mot_actu_plus3[3]] +=1

						if(mot_actu_plus3[1] == "et"  or mot_actu_plus3[1] == "ou"):
							if(mot_actu_plus2[3] == "ADJ" and mot_actu_plus4[3] == "ADJ"):
							#on ajoute dans dicocat et dico word les binomiaux que on a trouvé
								AllDictionnaryCat[mot_actu_plus2[3],mot_actu_plus3[3],mot_actu_plus4[3]] +=1
								AllDictionnaryWord[mot_actu_plus2[1]+"/"+mot_actu_plus2[3],mot_actu_plus3[1]+"/"+mot_actu_plus3[3],mot_actu_plus4[1]+"/"+mot_actu_plus4[3]] +=1

							if(mot_actu_plus2[3] == "NC" and mot_actu_plus4[3] == "NC"):
								AllDictionnaryCat[mot_actu_plus2[3],mot_actu_plus3[3],mot_actu_plus4[3]] +=1
								AllDictionnaryWord[mot_actu_plus2[1]+"/"+mot_actu_plus2[3],mot_actu_plus3[1]+"/"+mot_actu_plus3[3],mot_actu_plus4[1]+"/"+mot_actu_plus4[3]] +=1

							if(mot_actu_plus2[3] == "VPP" and mot_actu_plus4[3] == "VPP"):
								AllDictionnaryCat[mot_actu_plus2[3],mot_actu_plus3[3],mot_actu_plus4[3]] +=1
								AllDictionnaryWord[mot_actu_plus2[1]+"/"+mot_actu_plus2[3],mot_actu_plus3[1]+"/"+mot_actu_plus3[3],mot_actu_plus4[1]+"/"+mot_actu_plus4[3]] +=1
							
							if(mot_actu_plus2[3] == "V" and mot_actu_plus4[3] == "V"):
								AllDictionnaryCat[mot_actu_plus2[3],mot_actu_plus3[3],mot_actu_plus4[3]] +=1
								AllDictionnaryWord[mot_actu_plus2[1]+"/"+mot_actu_plus2[3],mot_actu_plus3[1]+"/"+mot_actu_plus3[3],mot_actu_plus4[1]+"/"+mot_actu_plus4[3]] +=1

					if(mot_actu_plus2[1] == "et" or mot_actu_plus2[1] == "ou"): #on regarde les binomiaux à 5 mots
						if(mot_actu[3] == "DET" and mot_actu_plus1[3] == "NC" and mot_actu_plus3[3] == "DET" and mot_actu_plus4[3] == "NC"):
							AllDictionnaryCat[mot_actu[3],mot_actu_plus1[3],mot_actu_plus2[3],mot_actu_plus3[3],mot_actu_plus4[3]]+=1
							AllDictionnaryWord[mot_actu[1]+"/"+mot_actu_plus1[3],mot_actu_plus1[1]+"/"+mot_actu_plus1[3],mot_actu_plus2[1]+"/"+mot_actu_plus2[3],mot_actu_plus3[1]+"/"+mot_actu_plus3[3],mot_actu_plus4[1]+"/"+mot_actu_plus4[3]]+=1

						if(mot_actu[3] == "P" and mot_actu_plus1[3] == "NC" and mot_actu_plus3[3] == "P" and mot_actu_plus4[3] == "NC"):
							AllDictionnaryCat[mot_actu[3],mot_actu_plus1[3],mot_actu_plus2[3],mot_actu_plus3[3],mot_actu_plus4[3]]+=1
							AllDictionnaryWord[mot_actu[1]+"/"+mot_actu[3],mot_actu_plus1[1]+"/"+mot_actu_plus1[3],mot_actu_plus2[1]+"/"+mot_actu_plus2[3],mot_actu_plus3[1]+"/"+mot_actu_plus3[3],mot_actu_plus4[1]+"/"+mot_actu_plus4[3]]+=1
	
	titre = 'Dico_Sub_words.csv'
	w1 = csv.writer(open(titre,'w',encoding="UTF-8"))
	for clef_mot, occu_mot in AllDictionnaryWord.items():
		if(occu_mot != 3):
			w1.writerow([clef_mot,occu_mot])

	titre = 'Dico_Sub_cats.csv'
	w2 = csv.writer(open(titre,'w',encoding="UTF-8"))
	for clef_cat, occu_cat in AllDictionnaryCat.items():
			w2.writerow([clef_cat,occu_cat])

	return (AllDictionnaryWord, AllDictionnaryCat)

def rep_graphiques(tous_les_dicos_word, tous_les_dicos_cat):
	"""Fonction qui établis graphiquement le nombre d'occurences des catégories et la quantité de binomes dans les corpus
	les sauvegardes dans le dossier où se situe le fichier d'exécution python.

    :param: 2 dictionnaires de mots et de catégories
    :type : defaultdict(int)
    """
	
	d = {'corpus': ["AA", "subtitles"], 'sommes': [sum(compte for binome, compte in tous_les_dicos_word[0].items() if compte <= 3 or compte >= 200), 
													sum(compte for binome, compte in tous_les_dicos_word[1].items() if compte <= 3 or compte >=200)]}
	data = pd.DataFrame(d)
	ax = sns.barplot(x = "corpus", y = "sommes", data = data)
	ax.set_ylabel("#binomes")
	plt.savefig("nbr_de_binomes")
	plt.close()

	data = [] 
	somme_AA = sum(compte for binome, compte in tous_les_dicos_cat[0].items())
	somme_sub = sum(compte for binome, compte in tous_les_dicos_cat[1].items())
	
	data.extend({"binomes": binome, "proba": compte/somme_AA, 'corpus': "AA"}
				for binome, compte in tous_les_dicos_cat[0].items())
	data.extend({"binomes": binome, "proba": compte/somme_sub, 'corpus': "subtitles"}
				for binome, compte in tous_les_dicos_cat[1].items())
	data = pd.DataFrame(data)
	ax = sns.barplot(x = "binomes", y = "proba", hue = "corpus", data = data)
	ax.figure.subplots_adjust(bottom = 0.40)

	for item in ax.get_xticklabels():
		item.set_rotation(45)
	
	ax.set_ylabel("probabilites")
	plt.savefig("distrib_corpus_cat")
	plt.close()

def freq_ordre(dico_word):
	"""Fonction qui établit un dictionnaire qui dit si un binome est en ordre alphabitique et/ou figé

    :param: dictionnaire de mots
    :type : defaultdict(int)
    :return: Un dictionnaire qui comprends un binomes en clef et en valeurs un tuple (alpha,ordre)
    :rtype: defaultdict(int)
    """
	dico_ordre = defaultdict(int)

	for binome in dico_word:
		if len(binome) == 5:
			if binome[0][3:] or binome[3][3:] == "/DET":
				dico_ordre[binome] = (ordre_abc(binome[1], binome[4]), ordre_fige(dico_word, binome))
		else:
			dico_ordre[binome] = (ordre_abc(binome[0], binome[2]), ordre_fige(dico_word, binome))
	return dico_ordre

def ordre_abc(s1, s2):
	"""Fonction retourne vrai si s1 est avant s2 dans l'alphabet

    :param: 2 Strings que l'on compare
    :type : String
    :return: 1 si c'est dans l'ordre alphabétique sinon 0
    :rtype: int
    """
	if s1[0]<s2[0]: return 1
	return 0

def ordre_fige(dico_word, binome):
	"""Fonction la proportionnalité entre 2 binomes inversé si existe sinon 0

    :param: un dictionnaire de mot et un binome
    :type : defaultdict(int), tuple
    :return: une proportion 
    :rtype: int
    """
	nv_binome = ()

	if len(binome) == 5:
		if binome[0][3:] or binome[3][3:] == "/DET": # + P
			nv_binome = (binome[0], binome[4], binome[2], binome[3], binome[1]) #on inverse
	else: #len(3)
		nv_binome = (binome[2], binome[1], binome[0])

	if nv_binome in dico_word:
		#p-s/s-p = 1 -> on a autant de forme p-s que s-p arbi
		#p-s/s-p > 1 p-s est plus fréquent que s-p
		#p-s/s-p < 1 s-p est plus fréquent que p-s
		#le resultat fait reférence à une proportion entre les 2 binomes
		return float(dico_word[binome]/(dico_word[nv_binome]+dico_word[binome]))
	return 0

def liste_freq_ordre(dico_ordre):
	"""Fonction qui retourne une liste avec le nombre d'occurences des fréquences d'ordre

    :param: un dictionnaire d'ordre
    :type : defaultdict(int)
    :return: une liste des occurences des fréquences d'ordre
    :rtype: list
    """
	liste_ordre = []
	binomes_fige =0
	binomes_arbi =0
	binomes_alpha_fige =0
	binomes_alpha_arbi =0
	binomes_N_alpha_fige =0
	binomes_N_alpha_arbi =0
	for binomes in dico_ordre.values():
		#print(binomes)
		alpha, fige_ou_N = binomes

		if(fige_ou_N >= 0.7): #0.7 car on considère ce seuil pour qu'il soit figé
			binomes_fige+=1
		elif(fige_ou_N < 0.7):
			binomes_arbi+=1

		if(alpha == 1 and fige_ou_N >= 0.7):
			binomes_alpha_fige+=1
		elif(alpha == 1 and fige_ou_N < 0.7):
			binomes_alpha_arbi +=1

		if(alpha == 0 and fige_ou_N >= 0.7):
			binomes_N_alpha_fige+=1
		elif(alpha == 0 and fige_ou_N < 0.7):
			binomes_N_alpha_arbi+=1

	sommeT = binomes_fige + binomes_arbi +binomes_alpha_fige + binomes_alpha_arbi + binomes_N_alpha_fige + binomes_N_alpha_arbi
	liste_ordre.append(binomes_fige/sommeT)
	liste_ordre.append(binomes_arbi/sommeT)
	liste_ordre.append(binomes_alpha_fige/sommeT)
	liste_ordre.append(binomes_alpha_arbi/sommeT)
	liste_ordre.append(binomes_N_alpha_fige/sommeT)
	liste_ordre.append(binomes_N_alpha_arbi/sommeT)

	return liste_ordre

def create_dico_obs():
	"""Fonction qui créer un dico_d'observation

    :return: un dictionnaire d'observations
    :rtype: default(int)
    """
	dico_obs = defaultdict(int)
	for binome in data_sem:
		obs = get_features(binome)
		dico_obs[binome[0] + " " + data_sem[binome]["kind"][0] + " " + binome[1]] = obs
	return dico_obs

def repr_graph_liste_ordre(liste_ordre_AA,liste_ordre_sub):
	"""Fonction qui créer un graphique avec les occurences des fréquences d'ordre et qui l'enregistre
	dans le dossier contenant le fichier python exécutable

	:param: 2 listes d'ordres (wiki et sous-titres)
    :type : list
    """
	n_grp = 6
	bar_width = 0.40
	index = np.arange(n_grp)
	rect1 = plt.bar(index,liste_ordre_AA,bar_width,label='AA')
	rect2 = plt.bar(index + bar_width,liste_ordre_sub,bar_width,color ='r',label = 'subtitles')
	plt.xlabel("Type d'ordre")
	plt.ylabel("probabilites")
	plt.xticks(index+ bar_width,('fige','arbi','alpha_fige','alpha_arbi','N_alpha_fige','N_alpha_arbi'),rotation = 45)
	plt.legend()
	plt.tight_layout()
	plt.savefig("distrib_ordre_binomes")
	plt.close()

def get_features(binome):
	"""Creer un dico à partir d'un binome

	:param: un binome
    :type : tuple
    :return: un dictionnaire d'observations
    :rtype: default(int)
    """
	obs = defaultdict(lambda: defaultdict(int))
	#rajouter le nb d'occurrence des deux elt

	mot_a1 = binome[0]
	mot_a2 = binome[1]

	#on cherche a ajouter la phonétique
	if mot_a1 in data_phon:
		obs[mot_a1]["phonetic"] = data_phon.get(mot_a1)
		#obs[mot_a1]["nb_syllabes"] = len(obs[mot_a1]["phonetic"][0].split("."))
	if mot_a2 in data_phon:
		obs[mot_a2]["phonetic"] = data_phon.get(mot_a2)
		#obs[mot_a2]["nb_syllabes"] = len(obs[mot_a2]["phonetic"][0].split("."))
	if "phonetic" in obs[mot_a1] and "phonetic" in obs[mot_a2]:
		if len(obs[mot_a1]["phonetic"])>0 and len(obs[mot_a2]["phonetic"])>0:
			obs["m1 + syll. m2"] = "yes" if len(obs[mot_a1]["phonetic"][0].split(".")) > len(obs[mot_a2]["phonetic"][0].split(".")) else "no"
	
	#on ajoute les critères sémantiques
	#cle qu'on rejoutera après
	a_eviter = mot_a1 + "  est plus générale que  " + mot_a2
	 
	for feat in data_sem[binome]["features"]:
		if feat != a_eviter:
			for cle in data_sem[binome]["features"][feat]:
				obs[feat][cle] = data_sem[binome]["features"][feat][cle]["meta"]

	if "meta" in data_sem[binome]["features"][a_eviter]:
		obs["m1 + gener. m2 "] = data_sem[binome]["features"][a_eviter]["meta"]
	#obs["kind"] = data_sem[binome]["kind"] #on ne prends pas en compte ici kind
	return obs

def liste_param_freq(dico_binomes_param):
	"""Creer des vecteurs d'observations des binomes d'une dictionnaire et les enregistre dans un fichier.txt
	Renvoies aussi 2 dictionnaires des occurences des parametres en fonction de leurs valeurs

	:param: un dictionnaire de binomes ayant pour valeurs des observations
    :type : defaultdict(int)
    :return: 2 dictionnaires d'occurences des observations
    :rtype: default(int)
    """
	d_freq_para_yes = defaultdict(int)
	d_freq_para_no = defaultdict(int)
	all_yes = 0
	all_no = 0
	liste_all_parametre = []
	for mots in dico_binomes_param:
		liste_param_mot = []
		liste_param_mot.append(mots)
		for params in (dico_binomes_param[mots]):
			if isinstance (dico_binomes_param[mots][params],dict):
				for features in dico_binomes_param[mots][params]:
					if features != "phonetic": #on ne prends pas la clef phonetic
						if dico_binomes_param[mots][params][features] == "yes":
							all_yes +=1
							liste_param_mot.append(1)
							d_freq_para_yes[features]+=1
						else:
							all_no +=1
							liste_param_mot.append(-1)
							d_freq_para_no[features]+=1
			if isinstance(dico_binomes_param[mots][params],str) and dico_binomes_param[mots][params] == "yes":
				all_yes +=1
				liste_param_mot.append(1)
				d_freq_para_yes[params]+=1
			elif isinstance(dico_binomes_param[mots][params],str) and dico_binomes_param[mots][params] == "no":
				all_no +=1
				liste_param_mot.append(-1)
				d_freq_para_no[params]+=1

		liste_all_parametre.append(liste_param_mot)
		

	with open("Data_features.txt","w",encoding="UTF-8") as f:
		for params in liste_all_parametre:
			a = str(params).replace("['","")
			b = str(a).replace("]","")
			c = str(b).replace("'","")
			f.write("%s\n" % c)
			
	d_freq_para_yes = {k : v/all_yes for k, v in d_freq_para_yes.items()}
	d_freq_para_no = {k : v/all_no for k, v in d_freq_para_no.items()}
	
	return d_freq_para_yes, d_freq_para_no

def repr_graph_features(dico_binomes_feats_yes, dico_binomes_feats_no):
	"""Creer 2 graphiques représentant les occurences des observations et les sauvegardes à l'endroit
	où se situe le fichier python d'exécution

	:param: 2 dictionnaires d'observation avec pour valeurs respectifs oui et non
    :type : defaultdict(int)
    """
	bar_width = 0.4

	sorted_yes = collections.OrderedDict(sorted(dico_binomes_feats_yes.items()))
	sorted_no = collections.OrderedDict(sorted(dico_binomes_feats_no.items()))

	rect1 = plt.bar(range(len(sorted_yes)),list(sorted_yes.values()),bar_width,label= 'features yes')
	plt.ylabel("probabilites")
	plt.xticks(range(len(sorted_yes)), list(sorted_yes.keys()),rotation = 45)
	plt.legend()
	plt.tight_layout()
	plt.savefig("distrib_feats_yes_binomes")
	plt.close()

	rect2 = plt.bar(range(len(sorted_no)),list(sorted_no.values()),bar_width,color = 'r',label ='features no')
	plt.ylabel("probabilites")
	plt.xticks(range(len(sorted_no)), list(sorted_no.keys()),rotation = 45)
	plt.legend()
	plt.tight_layout()
	plt.savefig("distrib_feats_no_binomes")
	plt.close()

##################### Etude de stats wiki et sub #########################
(dico_word_AA, dico_cat_AA) = lecture_wiki(allDocs)
(dico_word_sub, dico_cat_sub) = lecture_sub(allSubtitles)
rep_graphiques([dico_word_AA, dico_word_sub], [dico_cat_AA, dico_cat_sub])

dico_ordre_AA = freq_ordre(dico_word_AA)
liste_ordre_AA = liste_freq_ordre(dico_ordre_AA)
dico_ordre_sub = freq_ordre(dico_word_sub)
liste_ordre_sub = liste_freq_ordre(dico_ordre_sub)
repr_graph_liste_ordre(liste_ordre_AA,liste_ordre_sub)

##################### Corpus déja annoté #########################
dico_obs = create_dico_obs()
dico_freq_feats_yes, dico_freq_feats_no = liste_param_freq(dico_obs)
repr_graph_features(dico_freq_feats_yes, dico_freq_feats_no)


