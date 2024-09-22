import cv2
import numpy as np
import os

def preprocess_char_roi(roi):
    """对字符ROI进行预处理"""
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.equalizeHist(roi_gray)  # 直方图均衡化
    _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return roi_thresh

def match_template(roi, template, threshold=0.8):
    """在ROI中进行模板匹配，返回匹配结果的坐标和置信度"""
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    return loc

def process_image(img_path, template_folder):
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
    
    # 加载字符模板
    templates = {}
    for filename in os.listdir(template_folder):
        if filename.endswith('.png'):
            template_path = os.path.join(template_folder, filename)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            templates[filename] = template

    word_recognition = []  # 用于存储识别的字符及其位置

    for r in stats:
        x, y, w, h, s = r.tolist()
        if w * 1.5 < h and w * 10 > h and h > threshold.shape[0] / 2:
            word_rect.append(r)
            cv2.rectangle(ROI, (x, y), (x + w, y + h), (255, 0, 255), 2)

            # 进行模板匹配
            roi_char = ROI[y:y+h, x:x+w]
            roi_char_processed = preprocess_char_roi(roi_char)

            best_match_name = None
            best_match_score = -1  # 初始得分为-1

            for name, template in templates.items():
                # 调整模板大小以匹配字符区域
                template_resized = cv2.resize(template, (w, h))
                loc = match_template(roi_char_processed, template_resized)

                # 计算匹配得分
                if loc[0].size > 0:  # 如果有匹配
                    score = np.max(cv2.matchTemplate(roi_char_processed, template_resized, cv2.TM_CCOEFF_NORMED))
                    if score > best_match_score:
                        best_match_score = score
                        best_match_name = name

            # 如果找到最好的匹配，将其记录
            if best_match_name is not None:
                word_recognition.append((best_match_name, (x, y, w, h)))
                print(f'Matched {best_match_name} with score {best_match_score} at {(x, y)}')

    # 打印识别结果
    print("Recognized characters:", word_recognition)

    cv2.imwrite(img_path.replace('.png', '_words.png'), ROI)

if __name__ == '__main__':
    folder_path = r'F:\license_plate_recognition\license_plate_test'  # 替换为你的文件夹路径
    template_folder = r'F:\license_plate_recognition\template_source'  # 替换为你的模板文件夹路径
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            process_image(img_path, template_folder)
