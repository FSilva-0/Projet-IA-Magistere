import numpy as np

class IA:
    def __init__(self, dim_couche, tauxapp):
        self.nbcouche = len(dim_couche) - 1  # Nombre de transitions entre couches
        self.nbneuroneparcouche = dim_couche
        self.tauxapp = tauxapp
        self.poids = {}
        self.biais = {}
        self.cache = {}
        self.gradients = {}
        self.initialisation_parametres()

    def fonction_activation(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self,x):
        return np.exp(x)/sum(np.exp(x))

    def d_activation(self, x):  # Dérivée de la sigmoïde
        return x * (1 - x)

    def initialisation_parametres(self):
        for l in range(1, len(self.nbneuroneparcouche)):
            self.poids[f'W{l}'] = np.random.randn(self.nbneuroneparcouche[l], self.nbneuroneparcouche[l - 1]) * 0.1
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
            self.gradients[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True)

            if l > 1:
                W = self.poids[f'W{l}']
                dZ = np.dot(W.T, dZ) * self.d_activation(self.cache[f"A{l - 1}"])

        self.update_parameters()

    def update_parameters(self):
        for l in range(1, self.nbcouche + 1):
            self.poids[f'W{l}'] -= self.tauxapp * self.gradients[f'dW{l}']
            self.biais[f'b{l}'] -= self.tauxapp * self.gradients[f'db{l}']

    def training(self, df):
        data = df.values
        np.random.shuffle(data)
        tauxreussite = 0
        for i in range(len(data)):
            y_true = int(data[i,0])
            image = data[i,1:].reshape(-1, 1) / 255
            prediction = self.Forwardprop(image)
            if prediction == y_true:
                tauxreussite += 1
            self.Backwardprop(y_true)
            if i>0 and i%1000 == 0:
                print(f"Iteration {i} Precision cumulée: {(tauxreussite / i)*100:.2f}%")


    def predict(self, df):
        data = df.values
        tauxreussite = 0
        for i in range(len(data)):
            image = data[i,1:].reshape(-1, 1) / 255
            prediction = self.Forwardprop(image)
            y_true = int(data[i,0])
            if prediction == y_true:
                tauxreussite += 1
            if i>0 and i%1000 == 0:
                print((tauxreussite/i)*100)

