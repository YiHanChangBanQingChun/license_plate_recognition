import cv2
import numpy as np
import os

def preprocess_character(char_img):
    # 将字符图像转换为灰度图像
    gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    # 使用大津算法进行二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 调整图像大小为20x20
    resized = cv2.resize(binary, (20, 40), interpolation=cv2.INTER_AREA)
    return resized

def load_templates(template_folder):
    templates = {}
    # 遍历模板文件夹中的所有PNG文件
    for filename in os.listdir(template_folder):
        if filename.endswith('.png'):
            label = os.path.splitext(filename)[0]  # 使用文件名作为标签
            template_path = os.path.join(template_folder, filename)
            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template_img is None:
                print(f"Warning: Failed to load image {template_path}")
                continue
            # 使用大津算法进行二值化
            _, template_binary = cv2.threshold(template_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # 调整图像大小为20x20
            resized_template = cv2.resize(template_binary, (20, 40), interpolation=cv2.INTER_AREA)
            templates[label] = resized_template
    return templates

def match_character(char_img, templates):
    best_score = -1
    best_label = None
    # 遍历所有模板，找到匹配度最高的模板
    for label, template in templates.items():
        result = cv2.matchTemplate(char_img, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label

def save_image(image, output_folder, img_name, suffix):
    base, ext = os.path.splitext(img_name)
    save_path = os.path.join(output_folder, f"{base}_{suffix}{ext}")
    cv2.imwrite(save_path, image)
    print(f"Saved {save_path}")

def process_image(img_path, templates, output_folder):
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path, 1)

    # 检查图像的横向分辨率，如果小于5000，则放大到5000，纵向等比放大
    height, width = img.shape[:2]
    if width < 5000:
        scale_factor = 5000 / width
        new_width = 5000
        new_height = int(height * scale_factor)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        print(f"Image {img_name} resized to {new_width}x{new_height}")

    oriimg = img.copy()
    save_image(img, output_folder, img_name, 'original')

    # 高斯模糊处理
    img = cv2.GaussianBlur(img, (5, 5), 0)
    save_image(img, output_folder, img_name, 'blurred')

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    save_image(hsv, output_folder, img_name, 'hsv')

    lower = np.array([100, 100, 100])
    upper = np.array([140, 255, 255])
    # 创建掩膜，只保留蓝色部分
    mask = cv2.inRange(hsv, lower, upper)
    save_image(mask, output_folder, img_name, 'mask')

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

    # 找到面积最大的矩形，假设为车牌
    max_rec = max(blocks, key=lambda a: a[4])
    cv2.rectangle(oriimg, (max_rec[0], max_rec[1]), (max_rec[0] + max_rec[2], max_rec[1] + max_rec[3]), (255, 0, 255), 2)
    save_image(oriimg, output_folder, img_name, 'detected_plate')

    # 提取车牌区域
    ROI = oriimg[max_rec[1]:max_rec[1] + max_rec[3], max_rec[0]:max_rec[0] + max_rec[2], :]
    save_image(ROI, output_folder, img_name, 'ROI')

    ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    save_image(ROI_gray, output_folder, img_name, 'ROI_gray')

    ret, threshold = cv2.threshold(ROI_gray, 0, 255, cv2.THRESH_OTSU)
    save_image(threshold, output_folder, img_name, 'threshold')

    # 清除边缘
    margin_size = 5
    threshold[0:margin_size, :] = 0
    threshold[-margin_size:, :] = 0
    threshold[:, 0:margin_size] = 0
    threshold[:, -margin_size:] = 0
    save_image(threshold, output_folder, img_name, 'threshold_cleared')

    # 膨胀和腐蚀操作
    kernel1 = np.ones((1, 3), dtype=np.uint8)
    dilate = cv2.dilate(threshold, kernel1, 1)
    save_image(dilate, output_folder, img_name, 'dilate1')

    kernel2 = np.ones((7, 1), dtype=np.uint8)
    dilate = cv2.dilate(dilate, kernel2, 1)
    save_image(dilate, output_folder, img_name, 'dilate2')

    erose = cv2.erode(dilate, kernel2, 3)
    save_image(erose, output_folder, img_name, 'erose1')

    # 清除边缘
    margin_width = int(0.001 * width)  # 根据横向像素长度的3%
    margin_height = int(0.03 * height)  # 根据纵向像素长度的3%

    erose[0:margin_height, :] = 0  # 上边
    erose[-margin_height:, :] = 0  # 下边
    erose[:, 0:margin_width] = 0  # 左边
    erose[:, -margin_width:] = 0  # 右边
    save_image(erose, output_folder, img_name, 'erose_cleared')

    erose2 = cv2.erode(erose, kernel2, 3)
    save_image(erose2, output_folder, img_name, 'erose2')

    # 连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erose, connectivity=4)
    word_rect = []
    for r in stats:
        x, y, w, h, s = r.tolist()
        if w * 1.5 < h and w * 10 > h and h > threshold.shape[0] / 2:
            word_rect.append(r)
            cv2.rectangle(ROI, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # 按照 x 坐标对字符区域进行排序
    word_rect.sort(key=lambda r: r[0])

    save_image(ROI, output_folder, img_name, 'detected_characters')

    recognized_text = ""
    for rect in word_rect:
        x, y, w, h, s = rect
        char_img = ROI[y:y + h, x:x + w]
        preprocessed_char = preprocess_character(char_img)
        label = match_character(preprocessed_char, templates)
        if label:
            recognized_text += label
            cv2.putText(ROI, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    save_image(ROI, output_folder, img_name, 'final_recognition')

    # 将识别结果保存到文件
    with open(os.path.join(output_folder, f"{img_name}_result.txt"), 'w') as f:
        f.write(f"Recognized Text: {recognized_text}")

    print(f"Recognized Text in {img_path}: {recognized_text}")

if __name__ == '__main__':
    folder_path = r'F:\license_plate_recognition\license_plate_test'  # 替换为你的文件夹路径
    template_folder = r'F:\license_plate_recognition\template_source'  # 替换为你的模板文件夹路径
    output_folder = r'F:\license_plate_recognition\output'  # 替换为你的输出文件夹路径

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    templates = load_templates(template_folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            process_image(img_path, templates, output_folder)