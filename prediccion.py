import cv2
from PIL import Image, ImageEnhance
import numpy as np
import tensorflow as tf

CATEGORIA = ["sana", "enfermo"]
tamaño_img = 720

def preparar(ruta):
    # Parámetros de edición
    brillo = 1.75  # Ajusta el brillo (1.0 = sin cambios)
    contraste = 1.05  # Ajusta el contraste (1.0 = sin cambios)
    altas_luces = 1.05  # Ajusta las altas luces (1.0 = sin cambios)
    sombras = 1.05  # Ajusta las sombras (1.0 = sin cambios)
    nitidez = 1.5  # Ajusta la nitidez (1.0 = sin cambios)

    imagen = Image.open(ruta)
    # Ajusta de parametros de la imagen
    ajuste_brillo = ImageEnhance.Brightness(imagen)
    imagen = ajuste_brillo.enhance(brillo)

    ajuste_contraste = ImageEnhance.Contrast(imagen)
    imagen = ajuste_contraste.enhance(contraste)

    ajuste_altasluces = ImageEnhance.Brightness(imagen)
    imagen = ajuste_altasluces.enhance(altas_luces)

    ajuste_sombra = ImageEnhance.Brightness(imagen)
    imagen = ajuste_sombra.enhance(sombras)

    ajuste_nitidez = ImageEnhance.Sharpness(imagen)
    imagen = ajuste_nitidez.enhance(nitidez)

    # Convertir la imagen a formato OpenCV a escala de grises
    imagen_cv = cv2.cvtColor(np.array(imagen), cv2.COLOR_BGR2GRAY)

    # Homogenizar tamaño de imagenes
    imagen_cv = cv2.resize(imagen_cv, (tamaño_img, tamaño_img))

    # Aplicar filtro de Sobel
    bordes_verticales = cv2.Sobel(imagen_cv, cv2.CV_8U, 1, 0, ksize=3)
    bordes_horizontales = cv2.Sobel(imagen_cv, cv2.CV_8U, 0, 1, ksize=3)
    gradiente = cv2.addWeighted(bordes_verticales, 0.5, bordes_horizontales, 0.5, 0)
    return gradiente.reshape(-1, tamaño_img, tamaño_img, 1)

def predecir():
    pred = tf.keras.models.load_model("modelos/cnn_model_neu32.keras")
    print(CATEGORIA[int(pred.predict([preparar(r'C:\Users\aguir\Documentos\pruebas de img\AJUSTE\edit_hpf_IMG_20240531_142018.jpg')])[0][0])])

predecir()