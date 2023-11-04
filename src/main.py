import cv2
import numpy as np
from pathlib import Path
from imutils.perspective import four_point_transform


def main():
    img = cv2.imread("imagenes\\foto5.jpg")

    # Obtenemos el contorno.
    document_contour = scanDetection(img)

    # Recortamos la imagen.
    warped = four_point_transform(img.copy(), document_contour.reshape(4, 2))
    cv2.imshow("Warped", warped)

    # Procesamos la imagen.
    processed = image_processing(warped)
    # Esto quita las líneas negras y tal. Podemos modficarlo en un futuro.
    # processed = processed[10 : processed.shape[0] - 10, 10 : processed.shape[1] - 10]
    cv2.imshow("Processed", processed)

    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    cv2.imshow("foto1", img)
    cv2.waitKey(0)

    cv2.imwrite("prueba1.jpg", processed)


def scanDetection(img):
    # Modificar esto según la resolución de la cámara.
    HEIGHT, WIDTH, _ = img.shape

    # Si no hay contorno, que pille toda la imagen.
    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])

    # Aplicamos los filtros.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Obtenemos los contornos y los ordenamos.
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    # Para cada contorno, obtenemos el área mayor.
    for contour in contours:
        # cv2.drawContours(img, contour, -1, (0, 255, 0), 3)

        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area:
                document_contour = approx
                max_area = area

    # Se dibuja el contorno.
    cv2.drawContours(img, [document_contour], -1, (0, 255, 0), 3)
    return document_contour


def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return threshold


if __name__ == "__main__":
    main()
