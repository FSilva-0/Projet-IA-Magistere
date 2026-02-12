from traitement_image import rgb_a_gris, binarisation, generar_imagen_rgb, importer_image
from decoupage import pre_normalisation, cadrage2, scanner_horizontal, scanner_vertical, cadrage, normaliser, \
    redimensionner, post_decoupage
import numpy as np
import matplotlib.pyplot as plt
from time import time
from IA import IA
import pandas as pd

#Traitement
imagen_test = importer_image("test.png")
imagen_gris = rgb_a_gris(imagen_test)
imagen_binaria = binarisation(imagen_gris, C=20,k=11)
plt.imshow(imagen_binaria, cmap="gray")
plt.show()
#Decoupage
mots_pre = cadrage2(imagen_binaria)
mot_propre = pre_normalisation(mots_pre)

#revision visuel
for i, mot in enumerate(mot_propre):
    print(f"Palabra {i}:")
    for lettre in mot:
        img_norm = normaliser(lettre)
        plt.imshow(img_norm, cmap='gray')
        plt.show()
