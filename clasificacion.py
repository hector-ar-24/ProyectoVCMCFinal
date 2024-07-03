import tkinter as tk
import os
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import tensorflow as tf

CATEGORIAS = ["ENFERMA", "SANA"]
tamaño_img = 1080

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
    ancho = gradiente.shape[1]
    alto = gradiente.shape[0]

    # Calcular las coordenadas del cropping
    cropping_width = 250
    cropping_height = 250
    x = int((ancho - cropping_width) / 2)
    y = int((alto - cropping_height) / 2)

    # Aplicar el cropping
    cropped_img = gradiente[y:y+cropping_height, x:x+cropping_width]
    return cropped_img.reshape(-1, 250, 250, 1)

def predecir(imagen):
    pred = tf.keras.models.load_model("modelos/cnn_model_neu64.keras")
    categoria = CATEGORIAS[int(pred.predict([imagen])[0][1])]
    return categoria

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Clasificador de Imágenes")

# Crear un canvas para mostrar la imagen
canvas = tk.Canvas(root, width=500, height=500)
canvas.pack(pady=20)

# Función para cargar y procesar la imagen
def cargar_imagen():
    ruta_imagen = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg;*.png;*.jpeg")])
    if ruta_imagen:
        nombre_imagen = os.path.basename(ruta_imagen)
        imagen = preparar(ruta_imagen)
        categoria = predecir(imagen)
        mostrar_resultado(nombre_imagen, categoria)
        mostrar_imagen(ruta_imagen)

# Función para mostrar el resultado
def mostrar_resultado(nombre_imagen, categoria):
    resultado_label.config(text=f"La imagen '{nombre_imagen}' es: {categoria}")

# Función para mostrar la imagen en el canvas
def mostrar_imagen(ruta_imagen):
    imagen = Image.open(ruta_imagen)
    imagen = imagen.resize((500, 500), resample=Image.BICUBIC)
    foto = ImageTk.PhotoImage(imagen)
    canvas.create_image(0, 0, anchor=tk.NW, image=foto)
    canvas.image = foto

# Botón para cargar la imagen
cargar_boton = tk.Button(root, text="Cargar Imagen", command=cargar_imagen)
cargar_boton.pack(pady=10)

# Etiqueta para mostrar el resultado
resultado_label = tk.Label(root, text="", font=("Arial", 16))
resultado_label.pack(pady=20)

root.mainloop()