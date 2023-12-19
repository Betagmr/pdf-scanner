import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from main import process_image, rotate_image
from pdf_parser import create_pdf_with_image

# Configuración de la página
st.set_page_config(
    page_title="Escáner de Imagen a PDF",
    page_icon=":camera:",
    layout="wide",
)

# Lista para almacenar las imágenes
temp_dir = Path().cwd() / "temp"


# Función para mostrar la interfaz de carga de imágenes
def upload_image(n_images):
    uploaded_file_list = st.file_uploader(
        "Subir una imagen",
        type=["jpg", "jpeg", "png"],
        key=None,
        accept_multiple_files=True,
    )

    if len(uploaded_file_list) > n_images:
        for uploaded_file in uploaded_file_list:
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)

                cv2.imwrite(str(temp_dir / f"foto{n_images + 1}.jpg"), image)
        st.rerun()


def create_sidebar(list_of_paths, list_of_images):
    # Barra lateral para cargar imágenes
    st.sidebar.header("Opciones")

    side_button = st.sidebar.button("Create PDF")

    st.sidebar.header("Page configuration")
    options = [f"Imagen {i + 1}" for i, _ in enumerate(list_of_images)]
    select_index = st.sidebar.selectbox(
        "Uploaded images",
        options=options,
        index=0,
    )

    if select_index is not None:
        image_index = options.index(select_index)

    side_rotation = st.sidebar.button("Rotar", key="Rotar")
    side_delete = st.sidebar.button("Borrar ", key="Borrar")

    if side_rotation and select_index is not None:
        img = list_of_images[image_index]
        img_path = list_of_paths[image_index]
        rotated_img = rotate_image(img)

        result = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, result)
        st.rerun()

    if side_delete and select_index is not None:
        img_path = list_of_paths[image_index]
        os.remove(str(img_path))
        st.rerun()

    if side_button:
        new_list = [process_image(i) for i in list_of_images]
        create_pdf_with_image(new_list, "result.pdf")


def load_image(image_path):
    image = cv2.imread(image_path)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Función principal de la aplicación
def main():
    st.title("Escáner de Imagen a PDF :camera: -> :book:")

    list_of_paths = [str(image_path) for image_path in temp_dir.glob("*")]
    list_of_images = [load_image(image_path) for image_path in list_of_paths]

    create_sidebar(list_of_paths, list_of_images)
    upload_image(len(list_of_images))

    # Mostrar imágenes subidas en el estado
    st.header("Imágenes Subidas")

    if len(list_of_images) > 0:
        for i, image in enumerate(list_of_images):
            st.image(
                image,
                caption=i + 1,
                width=500,
            )
    else:
        st.info("No se han subido imágenes todavía.")


if __name__ == "__main__":
    main()
