import numpy as np

#Valeur d'entrer [longueur,largeur]
x_entrer = np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[1,1.5]),dtype=float) 
#Valeur a deduire 
y = np.array(([1],[0],[1],[0],[1],[0],[1],[0]),dtype=float) #Données de sortie 1 = rouge / 0 = violet

#Les valeurs doivent etre entre 0 et 1
x_entrer = x_entrer/np.amax(x_entrer, axis=0)

#Recuperation des 8 premières valeurs qui nous interessent
X = np.split(x_entrer,[8])[0]
#Recuperation de la dernière valeur à deduire
xPrediction = np.split(x_entrer,[8])[1]

#Création de la classe de reseau neuronal 
class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2  #Nombre de neurone d'entrer
        self.outputSize = 1  #Nombre de neurone de sorti
        self.hiddenSize = 3 #Nombre de neurone caché
        
        #Génération aléatoire des poids
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  #Matrice 2X3
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  #Matrice 3X1
    
    #Fonction pour multipler nos valeur par le poids pour avoir la valeur finale
    def forward(self, X):
        self.z = np.dot(X,self.W1)  #Valeur d'entrer
        self.z2 = self.sigmoid(self.z)  #Valeur caché
        self.z3 = np.dot(self.z2, self.W2) #Valeur de sortie
        o = self.sigmoid(self.z3)
        return o
    
    #Fonction "sigmoïde" pour modéliser des probabilités
    def sigmoid(self,s):
        return 1/(1+np.exp(-s))
    
    #Dérivé de la prmière fonction sigmoïde
    def sigmoidPrime(self,s):
        return s * (1-s)
    
    #Fontion de rétropropagation
    def backward(self,X,y,o):
        self.o_error = y - o #Calcul de l'erreur
        self.o_delta = self.o_error * self.sigmoidPrime(o) #Calcul de l'erreur delta
        
        self.z2_error = self.o_delta.dot(self.W2.T)  #Calcul de l'erreur des neurones caché
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2) #Calcul de l'erreur delta des neurones caché
        
        #Mise a jour des poids
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)
        
    #Fonction pour mettre a jour les poids PLUSIEURS fois
    def train(self,X,y):
        o = self.forward(X)
        self.backward(X,y,o)
    
    #Fonction pour savoir ce que l'IA pense du resultat 
    def predict(self):
        print("Donnée prédite après entrainement: ")
        print("Entrée : \n" + str(xPrediction))
        print("Sortie : \n" + str(self.forward(xPrediction)))
        
        if(self.forward(xPrediction) < 0.5):
            print("La fleur est BLEU ! \n")
        else:
            print("La fleur est ROUGE ! \n")
            
#Création du reseau de neurones
NN = Neural_Network()

#Boucle pour repeter n fois la tache en mettant a jour les poids a chaque fois
for i in range(2000000):
    print('# ' + str(i) + "\n")
    print("Valeurs d'entrées: \n" + str(X))
    print("Sortie actuelle: \n" + str(y))
    print("Valeurs prédites: \n" + str(np.matrix.round(NN.forward(X),2)))
    print("\n")
    NN.train(X,y)

NN.predict()