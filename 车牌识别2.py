import cv2
import numpy as np
import os
import functools

# 创建输出文件夹
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_and_show(title, image, filename=None):
    """
    保存和显示图像。

    参数:
    - title: 窗口标题
    - image: 要显示的图像
    - filename: 要保存的文件名（可选）
    """
    cv2.imshow(title, image)
    if filename:
        cv2.imwrite(os.path.join(output_dir, filename), image)

def resize_plate(plate_img, target_width=1000):
    """
    调整车牌图像的尺寸，使其横向分辨率为target_width，同时保持纵横比。

    参数:
    - plate_img: 原始车牌图像
    - target_width: 目标宽度（默认1000像素）

    返回:
    - resized_plate: 调整后的车牌图像
    """
    original_height, original_width = plate_img.shape[:2]
    aspect_ratio = original_height / original_width
    target_height = int(target_width * aspect_ratio)
    resized_plate = cv2.resize(plate_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized_plate

def load_char_templates(template_dir, size=(20, 20)):
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            char = os.path.splitext(filename)[0]
            tmpl_path = os.path.join(template_dir, filename)
            if not os.path.isfile(tmpl_path):
                print(f"模板文件不存在: {tmpl_path}")
                continue

            tmpl = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
            if tmpl is None:
                print(f"无法读取模板文件: {tmpl_path}")
                continue

            _, tmpl_bin = cv2.threshold(tmpl, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            try:
                tmpl_resized = cv2.resize(tmpl_bin, size, interpolation=cv2.INTER_AREA)
                templates[char] = tmpl_resized
                print(f"模板已加载: {char}")
            except Exception as e:
                print(f"调整模板大小时出错: {tmpl_path}, 错误: {e}")
    return templates

def vertical_projection_split(char, min_gap_ratio=0.4):
    """
    使用垂直投影分割字符。

    参数:
    - char: 单个字符的二值图像
    - min_gap_ratio: 空隙的最小比例，用于确定分割阈值

    返回:
    - sub_chars: 分割后的子字符列表
    """
    vertical_sum = np.sum(char, axis=0) / 255
    threshold_val = np.max(vertical_sum) * min_gap_ratio
    split_points = []
    in_char = False
    start = 0
    for idx, val in enumerate(vertical_sum):
        if val > threshold_val and not in_char:
            in_char = True
            start = idx
        elif val <= threshold_val and in_char:
            in_char = False
            end = idx
            split_points.append((start, end))
    if in_char:
        split_points.append((start, len(vertical_sum) - 1))

    sub_chars = []
    for (start, end) in split_points:
        margin = 2  # 边缘扩展，避免字符边缘缺失
        start = max(start - margin, 0)
        end = min(end + margin, char.shape[1])
        sub_char = char[:, start:end]
        if sub_char.shape[1] > 5 and sub_char.shape[0] > 10:
            sub_chars.append(sub_char)
    return sub_chars

def recognize_character(char_img, templates):
    """
    识别单个字符。

    参数:
    - char_img: 单个字符的二值图像
    - templates: 字符模板字典

    返回:
    - best_char: 识别出的字符
    """
    best_score = -1
    best_char = None
    for char, tmpl in templates.items():
        # 调整字符图像和模板尺寸一致
        try:
            tmpl_resized = cv2.resize(tmpl, (char_img.shape[1], char_img.shape[0]), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"调整模板大小时出错: 字符 {char}, 错误: {e}")
            continue
        res = cv2.matchTemplate(char_img, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        if score > best_score:
            best_score = score
            best_char = char
    return best_char

def detect_license_plate(img, num, color_ranges=None):
    """
    检测车牌位置并返回车牌区域。

    参数:
    - img: 原始图像
    - num: 图像编号，用于保存文件
    - color_ranges: 颜色范围字典

    返回:
    - ROIs: 截取并调整尺寸后的车牌图像列表
    - bboxes: 车牌的外接矩形列表 (x, y, w, h)
    """
    # 高斯滤波去噪
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    save_and_show('Blurred Image', blurred, f'{num}_blurred.png')

    # RGB转HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 默认颜色范围（蓝色、白色、黄色）
    if color_ranges is None:
        color_ranges = {
            'blue': (np.array([100, 80, 80]), np.array([140, 255, 255])),
            'white': (np.array([0, 0, 200]), np.array([180, 30, 255])),
            'yellow': (np.array([20, 100, 100]), np.array([30, 255, 255]))
        }

    # 构建多颜色掩模
    masks = []
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        masks.append(mask)

    # 合并所有颜色的掩模
    combined_mask = functools.reduce(cv2.bitwise_or, masks)
    save_and_show('Combined Mask', combined_mask, f'{num}_combined_mask.png')

    # 形态学操作，去除噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    save_and_show('Mask after Morphology', combined_mask, f'{num}_mask_morphology.png')

    # 检测轮廓
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / h
        area = w * h

        # 根据车牌的长宽比和面积筛选轮廓
        if 2 < aspect_ratio < 6 and 500 < area < 50000:
            blocks.append(c)  # 保存轮廓而不是边界框

    if not blocks:
        print("未检测到车牌区域。")
        return [], []

    # 选择所有符合条件的车牌矩形
    plates = []
    for contour in blocks:
        # 获取轮廓的四个顶点
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # 只处理四边形轮廓
            plates.append(approx)

    # 截取所有车牌区域并调整尺寸
    ROIs = []
    bboxes = []
    for idx, plate in enumerate(plates):
        # 进行透视变换
        pts = plate.reshape(4, 2)
        dst = np.array([[0, 0], [1000, 0], [1000, 200], [0, 200]], dtype='float32')  # 目标矩形的四个点
        M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
        warped = cv2.warpPerspective(img, M, (1000, 200))

        ROIs.append(warped)
        bboxes.append(pts)  # 存储原始顶点

        # 保存每个车牌图像
        cv2.imwrite(os.path.join(output_dir, f'{num}_plate_{idx}.png'), warped)

    return ROIs, bboxes

def segment_characters(plate_img, num, templates=None):
    """
    分割车牌字符并保存每个字符的图像。
    
    参数:
    - plate_img: 截取的车牌图像
    - num: 图像编号，用于保存文件
    - templates: 字符模板字典（可选）
    
    返回:
    - refined_characters: 分割出的字符图像列表
    - plate_text: 识别出的车牌文本
    """
    # 转灰度
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    save_and_show('Plate Gray', gray, f'{num}_plate_gray.png')

    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    save_and_show('Plate Binary', binary, f'{num}_plate_binary.png')

    # 去除最边缘的一部分
    margin_size = 10
    binary[:margin_size, :] = 0
    binary[-margin_size:, :] = 0
    binary[:, :margin_size] = 0
    binary[:, -margin_size:] = 0
    save_and_show('Binary after Margin Removal', binary, f'{num}_binary_margin_removed.png')

    # 形态学操作，使用开操作去除噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 使用较大的核
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    save_and_show('Opening Image', opening, f'{num}_opening.png')

    # 膨胀操作，连接字符内的断裂
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 增加核大小
    dilated = cv2.dilate(opening, kernel_dilate, iterations=2)
    save_and_show('Dilated Image', dilated, f'{num}_dilated.png')

    # 腐蚀操作，防止字符之间连在一起
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 可以保留
    eroded = cv2.erode(dilated, kernel_erode, iterations=2)  # 减少迭代次数
    save_and_show('Eroded Image', eroded, f'{num}_eroded.png')


    # 连通域检测
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)

    # 获取车牌图像尺寸
    plate_height, plate_width = plate_img.shape[:2]
    plate_area = plate_width * plate_height

    # 动态计算面积阈值
    min_area = plate_area * 0.02  # 2%
    max_area = plate_area * 0.06  # 6%

    # 筛选字符区域
    characters = []
    char_img = plate_img.copy()
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = w / h
        # 动态调整字符筛选参数
        if 0.3 < aspect_ratio < 0.8 and min_area < area < max_area:
            cv2.rectangle(char_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            char = binary[y:y+h, x:x+w]
            characters.append((x, char))
            # 保存字符图像
            cv2.imwrite(os.path.join(output_dir, f'{num}_char_{i}.png'), char)

    save_and_show('Detected Characters', char_img, f'{num}_detected_characters.png')

    if not characters:
        print("未检测到字符。")
        return [], ""

    # 根据x坐标排序字符，确保顺序正确
    characters = sorted(characters, key=lambda item: item[0])
    sorted_characters = [char for (x, char) in characters]

    # 检查是否有合并的字符区域，根据宽度是否异常大来决定是否需要分割
    refined_characters = []
    for i, char in enumerate(sorted_characters):
        char_h, char_w = char.shape
        if char_w > char_h * 1.0:  # 如果字符宽度大于高度，可能包含多个字符
            # 使用垂直投影分割字符
            sub_chars = vertical_projection_split(char, min_gap_ratio=0.5)
            if sub_chars:
                for j, sub_char in enumerate(sub_chars):
                    # 调整子字符图像尺寸
                    sub_char_resized = cv2.resize(sub_char, (20, 40), interpolation=cv2.INTER_AREA)
                    refined_characters.append(sub_char_resized)
                    cv2.imwrite(os.path.join(output_dir, f'{num}_char_split_{i}_{j}.png'), sub_char_resized)
        else:
            # 调整字符图像尺寸
            char_resized = cv2.resize(char, (20, 40), interpolation=cv2.INTER_AREA)
            refined_characters.append(char_resized)
            cv2.imwrite(os.path.join(output_dir, f'{num}_char_resized_{i}.png'), char_resized)

    # 可视化细化后的字符
    if refined_characters:
        for idx, sub_char in enumerate(refined_characters):
            cv2.imshow(f'Character {idx}', sub_char)
            cv2.imwrite(os.path.join(output_dir, f'{num}_refined_char_{idx}.png'), sub_char)

    save_and_show('Refined Characters', char_img, f'{num}_refined_characters.png')

    # 字符识别（可选）
    plate_text = ""
    if templates:
        for idx, char in enumerate(refined_characters):
            recognized_char = recognize_character(char, templates)
            if recognized_char:
                plate_text += recognized_char
            else:
                plate_text += '?'
            # 可视化识别结果
            cv2.putText(char_img, recognized_char if recognized_char else '?', (10, 30 + idx*30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 保存识别结果图像
    if templates:
        cv2.imwrite(os.path.join(output_dir, f'{num}_recognized_plate.png'), char_img)
        save_and_show('Recognized Plate Text', char_img, f'{num}_recognized_plate.png')

    return refined_characters, plate_text

def main():
    """
    主函数，进行车牌识别。
    """
    input_dir = r"F:\license_plate_recognition\license_plate_test"        # 替换为您的输入图片文件夹路径
    template_dir = r"F:\license_plate_recognition\template_source"          # 替换为您的字符模板文件夹路径

    # 加载字符模板
    templates = load_char_templates(template_dir)
    if not templates:
        print("未加载到任何字符模板。请准备字符模板后重试。")
        return

    # 获取文件夹中的所有图像文件
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]

    if not image_files:
        print("输入文件夹中没有支持的图片格式。")
        return

    results = []

    for image_file in image_files:
        img_path = os.path.join(input_dir, image_file)
        print(f"正在处理图片: {image_file}")

        # 确保文件路径的编码正确
        img_path = os.path.abspath(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"无法读取图像 {img_path}，跳过。")
            continue

        # 检测车牌
        ROIs, bboxes = detect_license_plate(img, image_file)
        if not ROIs:
            results.append((image_file, "未检测到车牌区域"))
            continue

        for idx, (ROI, bbox) in enumerate(zip(ROIs, bboxes)):
            # 可选：车牌透视变换校正（如有需要）
            # ROI = correct_perspective(ROI, bbox)

            characters, plate_text = segment_characters(ROI, f"{image_file}_{idx}", templates=templates)
            if not characters:
                results.append((f"{image_file}_{idx}", "未检测到字符"))
                continue

            # 将识别结果保存到列表中
            results.append((f"{image_file}_{idx}", plate_text))

        # 等待按键以查看结果
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    # 打印识别结果
    for image_file, result in results:
        print(f"{image_file}: {result}")

    # 可选：将结果保存到文件
    with open('识别结果.txt', 'w', encoding='utf-8') as f:
        for image_file, result in results:
            f.write(f"{image_file}: {result}\n")

if __name__ == '__main__':
    main()
