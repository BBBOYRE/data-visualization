import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 🌟 终极参数字典：融合了你亲手调优的“黄金参数”
IMAGE_PARAMS = {
    # cell00: 手绘图。关闭 CLAHE，使用大津法自动找阈值。
    "cell00.jpg": {
        "use_clahe": False, 
        "method": "otsu", "thresh_val": 0, 
        "min_area": 15, "max_area": 5000, 
        "morph_op": "open", "morph_iter": 1
    },
    # cell01: 洋葱表皮。采用你实测的完美参数！精确过滤细胞壁，只留细胞核。
    "cell01.jpg": {
        "use_clahe": False, 
        "method": "fixed", "thresh_val": 94,   # 你调出的硬阈值
        "min_area": 43, "max_area": 1693,      # 精确的面积过滤范围
        "morph_op": "open", "morph_iter": 3    # 3次强力开运算熔断细胞壁
    },
    # cell02: 真实红细胞。开启 CLAHE 解决光照不均，使用局部自适应阈值。
    "cell02.jpg": {
        "use_clahe": True,  
        "method": "adaptive", "thresh_val": 0, 
        "min_area": 25, "max_area": 1000, 
        "morph_op": "close_open", "morph_iter": 1
    }
}

def count_cells(image_path, filename):
    if not os.path.exists(image_path):
        print(f"❌ 找不到图片: {image_path}")
        return None, None, 0

    # 1. 安全读取含中文路径的图片
    img_data = np.fromfile(image_path, dtype=np.uint8)
    original_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if original_img is None: 
        print(f"❌ 无法解析图片: {image_path}")
        return None, None, 0
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 2. 灰度化
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # 3. 获取专属参数，并决定是否使用 CLAHE 增强对比度
    params = IMAGE_PARAMS.get(filename, IMAGE_PARAMS["cell00.jpg"])
    if params["use_clahe"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_img = clahe.apply(gray_img)
        
    # 4. 高斯去噪
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # 5. 因地制宜的二值化策略
    if params["method"] == "otsu":
        ret, thresh_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif params["method"] == "fixed":
        ret, thresh_img = cv2.threshold(blurred_img, params["thresh_val"], 255, cv2.THRESH_BINARY_INV)
    else: # adaptive
        thresh_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    
    # 6. 形态学处理 (根据参数决定开运算还是闭-开运算)
    kernel = np.ones((3, 3), np.uint8)
    if params["morph_iter"] > 0:
        if params["morph_op"] == "open":
            morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=params["morph_iter"])
        elif params["morph_op"] == "close_open": 
            morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=params["morph_iter"])
            morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, kernel, iterations=params["morph_iter"])
    else:
        morph_img = thresh_img
        
    # 7. 轮廓提取
    contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = original_rgb.copy()
    cell_count = 0
    
    # 8. 面积过滤与结果标注
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if params["min_area"] < area < params["max_area"]:
            cell_count += 1
            cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2) # 画绿圈
            
            # 计算质心并标上数字
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.putText(result_img, str(cell_count), (cX - 10, cY + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return original_rgb, result_img, cell_count

def main():
    base_path = "../期末报告图片素材/"
    image_files = ["cell00.jpg", "cell01.jpg", "cell02.jpg"]
    
    for filename in image_files:
        full_path = os.path.join(base_path, filename)
        print(f"\n正在处理: {filename} ...")
        
        original, result, count = count_cells(full_path, filename)
        
        if original is not None:
            print(f"✅ {filename} 最终检测数量：{count}")
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original)
            plt.title(f"Original: {filename}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(result)
            plt.title(f"Result: {count} Targets Detected")
            plt.axis('off')
            
            plt.tight_layout()
            # 弹窗显示，关掉当前窗口才会处理下一张
            plt.show() 

if __name__ == "__main__":
    main()