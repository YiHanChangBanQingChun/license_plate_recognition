import cv2
import numpy as np
import os

def process_image(img_path):
    img = cv2.imread(img_path, 1)
    oriimg = img.copy()
    img = cv2.GaussianBlur(img, (5, 5), 0)
    oriimg2 = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([100, 100, 100])
    upper = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w > (h * 2)) and (w < (h * 6)):
            blocks.append([x, y, w, h, w * h])

    if not blocks:
        print(f"No license plate detected in {img_path}")
        return

    max_rec = max(blocks, key=lambda a: a[4])
    cv2.rectangle(oriimg, (max_rec[0], max_rec[1]), (max_rec[0] + max_rec[2], max_rec[1] + max_rec[3]), (255, 0, 255), 2)
    cv2.imwrite(img_path.replace('.png', '_plate.png'), oriimg)

    ROI = oriimg2[max_rec[1]:max_rec[1] + max_rec[3], max_rec[0]:max_rec[0] + max_rec[2], :]
    ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(ROI_gray, 0, 255, cv2.THRESH_OTSU)

    margin_size = 10
    threshold[0:margin_size, :] = 0
    threshold[-margin_size:, :] = 0
    threshold[:, 0:margin_size] = 0
    threshold[:, -margin_size:] = 0

    kernel1 = np.ones((1, 3), dtype=np.uint8)
    dilate = cv2.dilate(threshold, kernel1, 1)
    kernel2 = np.ones((7, 1), dtype=np.uint8)
    dilate = cv2.dilate(dilate, kernel2, 1)
    erose = cv2.erode(dilate, kernel2, 1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erose, connectivity=4)
    word_rect = []
    for r in stats:
        x, y, w, h, s = r.tolist()
        if w * 1.5 < h and w * 10 > h and h > threshold.shape[0] / 2:
            word_rect.append(r)
            cv2.rectangle(ROI, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cv2.imwrite(img_path.replace('.png', '_words.png'), ROI)

if __name__ == '__main__':
    folder_path = r'F:\license_plate_recognition\license_plate_test'  # 替换为你的文件夹路径
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            process_image(img_path)