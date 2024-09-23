import cv2
import numpy as np
import os

def preprocess_character(char_img):
    gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(binary, (20, 20), interpolation=cv2.INTER_AREA)
    return resized

def load_templates(template_folder):
    templates = {}
    for filename in os.listdir(template_folder):
        if filename.endswith('.png'):
            label = os.path.splitext(filename)[0]  # 文件名作为标签
            template_path = os.path.join(template_folder, filename)
            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            _, template_binary = cv2.threshold(template_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            resized_template = cv2.resize(template_binary, (20, 20), interpolation=cv2.INTER_AREA)
            templates[label] = resized_template
    return templates

def match_character(char_img, templates):
    best_score = -1
    best_label = None
    for label, template in templates.items():
        result = cv2.matchTemplate(char_img, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label

def save_image(image, img_path, suffix):
    """
    保存图像，在原文件名基础上添加后缀。
    """
    base, ext = os.path.splitext(img_path)
    save_path = f"{base}_{suffix}{ext}"
    cv2.imwrite(save_path, image)
    print(f"Saved {save_path}")

def process_image(img_path, templates):
    # 读取原始图像
    img = cv2.imread(img_path, 1)
    oriimg = img.copy()
    save_image(img, img_path, 'original')

    # 高斯模糊
    img = cv2.GaussianBlur(img, (5, 5), 0)
    save_image(img, img_path, 'blur')

    oriimg2 = img.copy()

    # 转换为HSV并进行颜色阈值处理
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([100, 100, 100])
    upper = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    save_image(mask, img_path, 'mask')

    # 查找轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w > (h * 2)) and (w < (h * 6)):
            blocks.append([x, y, w, h, w * h])

    if not blocks:
        print(f"No license plate detected in {img_path}")
        return

    # 绘制最大区域的矩形框（假设为车牌）
    max_rec = max(blocks, key=lambda a: a[4])
    cv2.rectangle(oriimg, (max_rec[0], max_rec[1]), 
                  (max_rec[0] + max_rec[2], max_rec[1] + max_rec[3]), 
                  (255, 0, 255), 2)
    save_image(oriimg, img_path, 'plate')

    # 提取车牌区域（ROI）
    ROI = oriimg2[max_rec[1]:max_rec[1] + max_rec[3], max_rec[0]:max_rec[0] + max_rec[2], :]
    save_image(ROI, img_path, 'roi')

    # 灰度化并阈值化
    ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(ROI_gray, 0, 255, cv2.THRESH_OTSU)
    save_image(threshold, img_path, 'threshold')

    # 去除边缘噪点
    margin_size = 10
    threshold_margin = threshold.copy()
    threshold_margin[0:margin_size, :] = 0
    threshold_margin[-margin_size:, :] = 0
    threshold_margin[:, 0:margin_size] = 0
    threshold_margin[:, -margin_size:] = 0
    save_image(threshold_margin, img_path, 'margin_removed')

    # 腐蚀和膨胀操作
    kernel1 = np.ones((1, 3), dtype=np.uint8)
    dilate1 = cv2.dilate(threshold_margin, kernel1, iterations=1)
    save_image(dilate1, img_path, 'dilate1')

    kernel2 = np.ones((7, 1), dtype=np.uint8)
    dilate2 = cv2.dilate(dilate1, kernel2, iterations=1)
    save_image(dilate2, img_path, 'dilate2')

    erose = cv2.erode(dilate2, kernel2, iterations=1)
    save_image(erose, img_path, 'erose')

    # 连接组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erose, connectivity=4)
    word_rect = []
    ROI_words = ROI.copy()
    for r in stats:
        x, y, w, h, s = r.tolist()
        if w * 1.5 < h and w * 10 > h and h > threshold.shape[0] / 2:
            word_rect.append(r)
            cv2.rectangle(ROI_words, (x, y), (x + w, y + h), (255, 0, 255), 2)
    save_image(ROI_words, img_path, 'words')

    # 识别字符
    recognized_text = ""
    ROI_recognized = ROI.copy()
    for rect in word_rect:
        x, y, w, h, s = rect
        char_img = ROI[y:y + h, x:x + w]
        preprocessed_char = preprocess_character(char_img)
        label = match_character(preprocessed_char, templates)
        if label:
            recognized_text += label
            # 在识别后的图像上标注识别结果
            cv2.putText(ROI_recognized, label, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    save_image(ROI_recognized, img_path, 'recognized')

    print(f"Recognized Text in {img_path}: {recognized_text}")

if __name__ == '__main__':
    folder_path = r'F:\license_plate_recognition\license_plate_test'  # 替换为你的文件夹路径
    template_folder = r'F:\license_plate_recognition\template_source'  # 替换为你的模板文件夹路径
    templates = load_templates(template_folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            process_image(img_path, templates)
