import numpy as np
import matplotlib.pyplot as plt


def generar_imagen_falsa():
    alto = 50
    ancho = 50
    imagen = np.zeros((alto, ancho), dtype=int)

    imagen[10:20, 5:7] = 1
    imagen[10:20, 10:15] = 1
    imagen[10:13, 5:15] = 1
    imagen[17:20, 5:15] = 1

    imagen[10:20, 25:35] = 1

    imagen[35:45, 5:45] = 1

    return imagen
img = generar_imagen_falsa()
plt.imshow(img, cmap='gray')
plt.title("Imagen Falsa: 0 (Negro/Fondo) - 1 (Blanco/Texto)")

def intervalles(liste):
    if not liste:
        return []
    intervalos = []
    start = liste[0]

    for k in range (1, len(liste)):
        if liste[k] != liste[k-1]+1:
            fin = liste[k-1]
            intervalos.append((start,fin))
            start = liste[k]
    intervalos.append((start,liste[-1]))
    return intervalos

def scanner_horizontal(image):
    res_x, res_y = image.shape
    somme = np.sum(image, axis=1)
    lignes = []
    for i in range(res_x):
        if somme[i] != 0:
            lignes.append(i)
    resultat = intervalles(lignes)
    return resultat

def scanner_vertical(image):
    res_x, res_y = image.shape
    somme = np.sum(image, axis=0)
    colonnes = []
    for i in range(res_y):
        if somme[i] != 0:
            colonnes.append(i)
    resultat = intervalles(colonnes)
    return resultat

def cadrage(image):
    caracteres = []
    lignes_avec_caracteres = scanner_horizontal(image)

    for (y_min, y_max) in lignes_avec_caracteres:
        image_ligne = image[y_min : y_max+1, :]
        colonnes_avec_caracteres = scanner_vertical(image_ligne)
        for (x_min, x_max) in colonnes_avec_caracteres:
            caractere = image_ligne[:, x_min : x_max+1]
            caracteres.append(caractere)
    return caracteres

def redimensionner(image)


print(scanner_horizontal(img))
print(scanner_vertical(img))
print(cadrage(img))
plt.show()
