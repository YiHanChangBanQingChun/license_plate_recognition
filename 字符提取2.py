import matplotlib.pyplot as plt
import os
import numpy as np
import imageio
import cv2

# 拼音字典
pinyin_dict = {
    "京": "jing", "津": "jin", "沪": "hu", "渝": "yu", "冀": "ji", "豫": "yu",
    "云": "yun", "辽": "liao", "黑": "hei", "湘": "xiang", "皖": "wan", "鲁": "lu",
    "新": "xin", "苏": "su", "浙": "zhe", "赣": "gan", "鄂": "e", "桂": "gui",
    "甘": "gan", "晋": "jin", "蒙": "meng", "陕": "shan", "吉": "ji", "闽": "min",
    "贵": "gui", "粤": "yue", "青": "qing", "藏": "zang", "川": "chuan", "宁": "ning", "琼": "qiong",
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G",
    "H": "H", "I": "I", "J": "J", "K": "K", "L": "L", "M": "M", "N": "N",
    "O": "O", "P": "P", "Q": "Q", "R": "R", "S": "S", "T": "T", "U": "U",
    "V": "V", "W": "W", "X": "X", "Y": "Y", "Z": "Z",
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "7", "8": "8", "9": "9"
}

def generate_character_image(char, font_size=10, image_size=(10, 20), font_family='SimHei', output_path='template_source'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 创建黑色背景的图形
    fig, ax = plt.subplots(figsize=(image_size[0]/100, image_size[1]/100), dpi=100, facecolor='black')
    ax.set_facecolor('black')
    
    # 添加白色字符
    ax.text(0.5, 0.5, char, fontsize=font_size, fontfamily=font_family,
            ha='center', va='center', color='white')  # 字体颜色设置为白色
    plt.axis('off')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout(pad=0)
    
    # 获取拼音名称
    pinyin_name = pinyin_dict.get(char, char)  # 获取拼音
    image_path = os.path.join(output_path, f"{pinyin_name}.png")
    
    # 保存图像，确保背景为黑色
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close(fig)
    
    # 将图像转换为灰度并二值化，直接覆盖原始 PNG 文件
    img = imageio.imread(image_path)
    gray = rgb_to_grayscale(img)
    binary = threshold(gray, thresh=128)
    
    # 裁剪图像，去除多余的黑色区域
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        print(f"Warning: No non-zero pixels found for character '{char}'")
        cropped = binary
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        # 可选：添加填充
        padding = 2  # 根据需要调整填充大小
        y_min = max(y_min - padding, 0)
        x_min = max(x_min - padding, 0)
        y_max = min(y_max + padding, binary.shape[0] - 1)
        x_max = min(x_max + padding, binary.shape[1] - 1)
        cropped = binary[y_min:y_max+1, x_min:x_max+1]
    
    # 调整大小到统一尺寸
    resized = resize(cropped, image_size)
    
    # 保存为二值化图像
    cv2.imwrite(image_path, resized)

def rgb_to_grayscale(img):
    if len(img.shape) == 3:
        grayscale = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        return grayscale.astype(np.uint8)
    else:
        return img

def threshold(image, thresh=128):
    binary = np.where(image > thresh, 255, 0).astype(np.uint8)
    return binary

def resize(image, size):
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized

def prepare_programmatic_templates(template_dir, chars, font_size=15, image_size=(10, 20), font_family='SimHei'):
    for char in chars:
        generate_character_image(char, font_size, image_size, font_family, template_dir)
        print(f"模板已生成: {char}")

if __name__ == "__main__":
    chinese_provinces = list("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼")
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = list("0123456789")
    characters = chinese_provinces + letters + digits

    template_dir = "template_source"
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    else:
        for file in os.listdir(template_dir):
            file_path = os.path.join(template_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    prepare_programmatic_templates(template_dir, characters, font_size=75, image_size=(50, 100), font_family='Microsoft YaHei')

    print("所有字符模板已生成并保存为PNG文件。")
