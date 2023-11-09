import cv2
import numpy as np
from pathlib import Path
from imutils.perspective import four_point_transform  # type: ignore


def main():
    img = cv2.imread("imagenes/foto4.jpg")

    # Obtenemos el contorno.
    document_contour = scanDetection(img)

    wraped = four_point_transform(img.copy(), document_contour.reshape(4, 2))
    processed = image_processing(wraped)

    print(processed.shape)
    print(processed)

    cv2.imshow("Processed", cv2.pyrDown(cv2.pyrDown(processed)))
    cv2.moveWindow("Processed", 1000, 1)

    # cv2.imshow("Wraped", cv2.pyrDown(cv2.pyrDown(wraped)))
    # cv2.moveWindow("Wraped", 1000, 0)

    img = cv2.pyrDown(img)
    # img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    cv2.imshow("foto1", img)
    cv2.moveWindow("foto1", 0, 1)
    cv2.waitKey(0)

    # cv2.imwrite("prueba1.jpg", processed)


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

    # Para cada contorno, obtenemos el área mayor.
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000 and area > max_area:
            document_contour = contour
            max_area = area

    # Se dibuja el contorno.
    rect = cv2.minAreaRect(document_contour)
    box = np.int0(cv2.boxPoints(rect))

    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    cv2.drawContours(img, [document_contour], -1, (255, 0, 0), 3)

    return box


def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return threshold


if __name__ == "__main__":
    main()
