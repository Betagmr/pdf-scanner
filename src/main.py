import cv2
import numpy as np
from pathlib import Path
from imutils.perspective import four_point_transform  # type: ignore


def main():
    image_path = Path() / "imagenes" / "foto5.jpg"
    img = cv2.imread(image_path.as_posix())

    # Obtenemos el contorno.
    pape_contour, document_contour = scanDetection(img)
    processed = image_processing(img, pape_contour, document_contour)

    cv2.imwrite("prueba1.jpg", processed)
    render_images(img, processed)


def render_images(img, processed):
    # Se muestra la imagen procesada
    cv2.imshow("Processed", img_downscale(processed, 2))
    cv2.moveWindow("Processed", 1000, 1)

    # Se muestra la imagen original
    cv2.imshow("foto1", img_downscale(img, 3))
    cv2.moveWindow("foto1", 0, 1)
    cv2.waitKey(0)


def get_img_contour(img):
    height, width, _ = img.shape
    document_contour = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    return document_contour


def img_downscale(img, scale=1):
    new_img = img.copy()
    for _ in range(scale):
        new_img = cv2.pyrDown(new_img)

    return new_img


def scanDetection(img):
    document_contour = get_img_contour(img)

    # Aplicamos los filtros.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Obtenemos los contornos y los ordenamos.
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Para cada contorno, obtenemos el área mayor.
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000 and area > max_area:
            document_contour = contour
            max_area = area

    # Se dibuja el contorno.
    rect = cv2.minAreaRect(document_contour)
    box = np.intp(cv2.boxPoints(rect))
    # cv2.drawContours(img, [document_contour], 0, (255, 255, 255), 100)

    return box, document_contour


def image_processing(image, page_contour, document_contour):
    aux_img = image.copy()
    cv2.fillPoly(aux_img, [page_contour], (255, 255, 255))
    cv2.fillPoly(aux_img, [document_contour], (0, 0, 0))
    # cv2.drawContours(aux_img, [document_contour], 0, (255, 255, 255), 150)

    result = cv2.add(aux_img, image)
    wrapped = four_point_transform(result.copy(), page_contour.reshape(4, 2))

    gray = cv2.cvtColor(wrapped, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return threshold


if __name__ == "__main__":
    main()
