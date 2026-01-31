from traitement_image import rgb_a_gris, binaire, generar_imagen_rgb
from decoupage import scanner_horizontal, scanner_vertical, cadrage, normaliser, redimensionner
import numpy as np
import matplotlib.pyplot as plt

imagen_a_color = generar_imagen_rgb()


plt.imshow(imagen_a_color)
plt.show()


imagen_gris = rgb_a_gris(imagen_a_color)
imagen_binaria = binaire(imagen_gris)
letra = cadrage(imagen_binaria)
for k in range(len(letra)):
    affichage = normaliser(letra[k])
    plt.imshow(affichage, cmap='gray')
    plt.show()