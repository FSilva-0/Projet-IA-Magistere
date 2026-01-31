import numpy as np
import matplotlib.pyplot as plt
import random as rd

def generar_imagen_rgb():
    alto = 50
    ancho = 100
    imagen = np.ones((alto, ancho, 3), dtype=int) * 255
    letra1= rd.randint(0,255)
    letra12 = rd.randint(0, 255)
    letra13 = rd.randint(0, 255)
    letra2= rd.randint(0,255)
    letra22 = rd.randint(0, 255)
    letra23 = rd.randint(0, 255)
    letra3= rd.randint(0,255)
    letra32 = rd.randint(0, 255)
    letra33 = rd.randint(0, 255)

    imagen[10:40, 10:15] = [letra1, letra12, letra13]
    imagen[10:15, 10:25] = [letra1, letra12, letra13]
    imagen[10:25, 20:25] = [letra1, letra12, letra13]
    imagen[20:25, 10:25] = [letra1, letra12, letra13]
    imagen[25:40, 20:25] = [letra1, letra12, letra13]
    imagen[10:40, 40:45] = [letra2, letra22, letra23]
    imagen[10:15, 40:55] = [letra2, letra22, letra23]
    imagen[35:40, 40:55] = [letra2, letra22, letra23]
    imagen[25:40, 50:55] = [letra2, letra22, letra23]
    imagen[10:40, 70:75] = [letra3, letra32, letra33]
    imagen[10:15, 70:85] = [letra3, letra32, letra33]
    imagen[20:25, 70:80] = [letra3, letra32, letra33]
    imagen[35:40, 70:85] = [letra3, letra32, letra33]

    return imagen

def rgb_a_gris(image):
    return np.sum(image, axis=2) / 3.0


def binaire(image):
    resultat = np.zeros_like(image)
    resultat[image < 200] = 1

    return resultat

