import cv2
import numpy as np
import os


def order_points(pts):

    rectangle = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rectangle[0] = pts[np.argmin(s)]
    rectangle[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rectangle[1] = pts[np.argmin(diff)]
    rectangle[3] = pts[np.argmax(diff)]
    return rectangle


def rectify_image(image, mask):

    shape, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    screenShape = max(shape, key=cv2.contourArea)
    perimeter = cv2.arcLength(screenShape, True)
    approx = cv2.approxPolyDP(screenShape, 0.02 * perimeter, True)

    if len(approx) != 4:
        x, y, w, h = cv2.boundingRect(screenShape)
        return image[y:y + h, x:x + w]

    points = approx.reshape(4, 2)
    rect = order_points(points)
    (topLeft, topRight, bottomRight, bottomLeft) = rect

    widthTop = np.sqrt(((bottomRight[0] - bottomLeft[0]) ** 2) + ((bottomRight[1] - bottomLeft[1]) ** 2))
    widthBottom = np.sqrt(((topRight[0] - topLeft[0]) ** 2) + ((topRight[1] - topLeft[1]) ** 2))
    maxWidth = max(int(widthTop), int(widthBottom))
    heightRight = np.sqrt(((topRight[0] - bottomRight[0]) ** 2) + ((topRight[1] - bottomRight[1]) ** 2))
    heightLeft = np.sqrt(((topLeft[0] - bottomLeft[0]) ** 2) + ((topLeft[1] - bottomLeft[1]) ** 2))
    maxHeight = max(int(heightRight), int(heightLeft))


    destination = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")


    matrix = cv2.getPerspectiveTransform(rect, destination)
    rectifiedImage = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

    return rectifiedImage

base_folder = r"C:\Users\amitw\OneDrive\Desktop\Tv_Dataset"
defected_folder = os.path.join(base_folder, "defected_tvs")
mask_folder = os.path.join(base_folder, "masks")
output_folder = os.path.join(base_folder, "rectified_training_data")

os.makedirs(output_folder, exist_ok=True)

for defect_type in os.listdir(defected_folder):
    input_path = os.path.join(defected_folder, defect_type)
    output_path = os.path.join(output_folder, defect_type)
    os.makedirs(output_path, exist_ok=True)

    if not os.path.isdir(input_path): continue

    for filename in os.listdir(input_path):
        if not filename.endswith(".png"): continue

        img = cv2.imread(os.path.join(input_path, filename))
        mask_path = os.path.join(mask_folder, filename.replace(".png", "_mask.png"))

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0)

            result = rectify_image(img, mask)

            if result is not None:
                cv2.imwrite(os.path.join(output_path, filename), result)
            else:
                print(f"Skipping {filename}, shape not found.")

