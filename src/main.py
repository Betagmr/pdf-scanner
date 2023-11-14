import cv2
import numpy as np
from pathlib import Path
from imutils.perspective import four_point_transform  # type: ignore


def main():
    for i in range(1, 6):
        image_path = Path() / "imagenes" / f"foto{i}.jpg"
        img = cv2.imread(image_path.as_posix())

        # Obtenemos el contorno.
        deleted_shadow = delete_shadow(img)
        pape_contour, document_contour = scanDetection(img)
        processed = image_processing(deleted_shadow, pape_contour, document_contour)

        render_images(img, processed)

    cv2.imwrite("prueba1.jpg", processed)


def main_camera():
    vid = cv2.VideoCapture(0)

    while True:
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        pape_contour, document_contour = scanDetection(frame)

        # render_images(frame, processed)
        cv2.imshow("frame", frame)
        cv2.moveWindow("frame", 0, 1)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            processed = image_processing(frame, pape_contour, document_contour)
            cv2.imwrite("prueba1.jpg", processed)

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def render_images(img, processed):
    # Se muestra la imagen procesada
    window_processed = "Processed"

    gray = 255 * (processed < 128).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    rect = processed[y : y + h, x : x + w]
    constant = cv2.copyMakeBorder(
        rect, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    cv2.imshow(window_processed, img_downscale(constant, 2))
    cv2.moveWindow(window_processed, 1000, 1)

    window_original = "Original"
    cv2.imshow(window_original, img_downscale(img, 2))
    cv2.moveWindow(window_original, 0, 1)

    cv2.waitKey(0)


def delete_shadow(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = [process_plane(plane) for plane in rgb_planes]
    return cv2.merge(result_norm_planes)


def process_plane(plane):
    dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(
        diff_img, None, alpha=0, beta=160, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )

    return norm_img


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

    cv2.drawContours(img, [box], 0, (0, 0, 255), 10)
    cv2.drawContours(img, [document_contour], 0, (255, 0, 0), 10)

    return box, document_contour


def image_processing(image, page_contour, document_contour):
    aux_img = image.copy()
    cv2.fillPoly(aux_img, [page_contour], (255, 255, 255))
    cv2.fillPoly(aux_img, [document_contour], (0, 0, 0))
    cv2.drawContours(aux_img, [document_contour], 0, (255, 255, 255), 100)

    result = cv2.add(aux_img, image)
    wrapped = four_point_transform(result.copy(), page_contour.reshape(4, 2))

    gray = cv2.cvtColor(wrapped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY)

    return threshold


if __name__ == "__main__":
    main()
