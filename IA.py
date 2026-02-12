import numpy as np
import pandas as pd


def charger_donnees():
    # Remplacez par vos chemins réels
    df_train = pd.read_csv('Magistère/emnist-byclass-train.csv', header=None)
    df_test = pd.read_csv('Magistère/emnist-byclass-test.csv', header=None)
    return df_train, df_test


class IA:
    def __init__(self, dim_couche, tauxapp):
        self.nbcouche = len(dim_couche) - 1
        self.nbneuroneparcouche = dim_couche
        self.tauxapp = tauxapp
        self.poids = {}
        self.biais = {}
        self.cache = {}
        self.gradients = {}
        self.initialisation_parametres()

    def fonction_activation(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=0)

    def d_activation(self, a):
        return a * (1 - a)

    def initialisation_parametres(self):
        for l in range(1, len(self.nbneuroneparcouche)):
            self.poids[f'W{l}'] = np.random.randn(self.nbneuroneparcouche[l], self.nbneuroneparcouche[l - 1]) * np.sqrt(
                2 / self.nbneuroneparcouche[l - 1])
            self.biais[f'b{l}'] = np.zeros((self.nbneuroneparcouche[l], 1))

    def Forwardprop(self, Image):
        self.cache["A0"] = Image.reshape(-1, 1)

        for l in range(1, self.nbcouche + 1):
            W = self.poids[f'W{l}']
            b = self.biais[f'b{l}']
            A_prev = self.cache[f"A{l - 1}"]
            Z = np.dot(W, A_prev) + b
            self.cache[f"Z{l}"] = Z

            if l == self.nbcouche:
                A = self.softmax(Z)
            else:
                A = self.fonction_activation(Z)
            self.cache[f"A{l}"] = A

        return np.argmax(self.cache[f"A{self.nbcouche}"])

    def Backwardprop(self, attendu):
        y = np.zeros((self.nbneuroneparcouche[-1], 1))
        y[attendu] = 1
        dZ = self.cache[f"A{self.nbcouche}"] - y

        for l in range(self.nbcouche, 0, -1):
            A_prev = self.cache[f"A{l - 1}"]
            self.gradients[f"dW{l}"] = np.dot(dZ, A_prev.T)
            self.gradients[f"db{l}"] = dZ

            if l > 1:
                W = self.poids[f'W{l}']
                # Rétropropagation à travers la sigmoïde de la couche précédente
                dZ = np.dot(W.T, dZ) * self.d_activation(self.cache[f"A{l - 1}"])

        self.update_parameters()

    def update_parameters(self):
        for l in range(1, self.nbcouche + 1):
            self.poids[f'W{l}'] -= self.tauxapp * self.gradients[f'dW{l}']
            self.biais[f'b{l}'] -= self.tauxapp * self.gradients[f'db{l}']

    def training(self, df):
        # Conversion en numpy pour la vitesse
        data = df.values
        np.random.shuffle(data)  # Mélanger les données est crucial

        tauxreussite = 0
        for i in range(len(data)):
            y_true = data[i, 0]
            image = data[i, 1:].reshape(-1, 1) / 255.0

            # Forward
            probabilites = self.Forwardprop(image)
            prediction = np.argmax(probabilites)

            if prediction == y_true:
                tauxreussite += 1

            # Backprop
            self.Backwardprop(y_true)

            if i > 0 and i % 1000 == 0:
                print(f"Iteration {i} | Précision cumulée: {(tauxreussite / i) * 100:.2f}%")

    def predict(self, df):
        data = df.values
        np.random.shuffle(data)
        tauxreussite = 0
        for i in range(len(data)):
            image = data[i, 1:].reshape(-1, 1) / 255.0
            prediction = self.Forwardprop(image)
            y_true = int(data[i, 0])
            if prediction == y_true:
                tauxreussite += 1
            if i > 0 and i % 1000 == 0:
                print(f"Iteration {i} | Précision cumulée: {(tauxreussite / i) * 100:.2f}%")

df_train, df_test = charger_donnees()
print(df_train.shape)
mon_ia = IA([784, 128,64, 62], 0.1)
mon_ia.training(df_train)
mon_ia.predict(df_test)


