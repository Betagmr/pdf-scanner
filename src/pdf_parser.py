from io import BytesIO

import cv2
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def create_pdf_with_image(image_list, output):
    # Obtenemos las dimensiones de la imagen.
    img_height, img_width = 1920, 1280

    # Creamos un lienzo.
    pdf_canvas = canvas.Canvas(output, pagesize=(img_width, img_height))

    # Convertimos la matriz NumPy en un objeto BytesIO.
    for image in image_list:
        image_stream = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_image.save(image_stream, format="PNG")
        image_stream.seek(0)

        # Dibujamos la imagen en el pdf.
        pdf_canvas.drawImage(
            ImageReader(image_stream), 0, 0, width=img_width, height=img_height
        )
        pdf_canvas.showPage()

    # Guardamos el pdf.
    pdf_canvas.save()
