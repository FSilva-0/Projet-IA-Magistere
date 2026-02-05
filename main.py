from traitement_image import rgb_a_gris, binarisation, generar_imagen_rgb, importer_image
from decoupage import pre_normalisation, cadrage2, scanner_horizontal, scanner_vertical, cadrage, normaliser, \
    redimensionner, post_decoupage
import numpy as np
import matplotlib.pyplot as plt
from time import time

image = importer_image("page.jpg")
im2 = rgb_a_gris(image)
debut = time()
im3 = binarisation(im2, k=11, C=20)
fin = time()
print(fin-debut)
plt.imshow(im3, cmap="gray")
plt.show()

"""
#Traitement
imagen_test = importer_image("test2.png")
imagen_gris = rgb_a_gris(imagen_test)
imagen_binaria = binaire(imagen_gris)

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
        """