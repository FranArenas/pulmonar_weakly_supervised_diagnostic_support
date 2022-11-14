import cv2


def check_file(filepath: str):
    image = cv2.imread(filepath)
    for row in image:
        for pixel in row:
            assert pixel[0] == pixel[1] == pixel[2]