import numpy as np
import matplotlib.pyplot as plt


def generar_imagen_falsa():
    alto = 50
    ancho = 50
    imagen = np.zeros((alto, ancho), dtype=int)

    imagen[10:20, 5:15] = 1

    imagen[10:20, 25:35] = 1

    imagen[35:45, 5:45] = 1

    return imagen
img = generar_imagen_falsa()
plt.imshow(img, cmap='gray')
plt.title("Imagen Falsa: 0 (Negro/Fondo) - 1 (Blanco/Texto)")

def scanner_horizontal(image):
    res_x, res_y = image.shape
    somme = np.sum(image, axis=1)
    lignes = []
    for i in range(res_y):
        if somme[i] != 0:
            lignes.append(i)
    return lignes



def scanner_vertical(image):
    res_x, res_y = image.shape
    somme = np.sum(image, axis=0)
    colonnes = []
    for i in range(res_x):
        if somme[i] != 0:
            colonnes.append(i)
    return colonnes

scanner_horizontal(img)
scanner_vertical(img)
plt.show()


def cadrage(image):
    liste_horizontale=scanner_horizontal(image)
    pixels=[]
    while len(liste_horizontale)!=0:
        liste=[]
        i=0
        while liste_horizontale[i]+1==liste_horizontale[i+1]:
            liste.append(i)
            liste_horizontale.remove(liste_horizontale[i])
            i+=1
        pixels.append(i+1)
        liste_horizontale.remove(liste_horizontale[i+1])
        pixels.append(liste)

