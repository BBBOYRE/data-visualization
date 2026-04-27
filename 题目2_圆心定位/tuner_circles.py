import cv2
import numpy as np
import os

def nothing(x): pass

def run_circle_tuner(image_filename):
    image_path = f"../期末报告图片素材/{image_filename}"
    img_data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"❌ 找不到图片或路径错误: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    cv2.namedWindow('Circle Tuner', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Circle Tuner', 800, 600) # 强制给一个比较大的初始窗口

    # 设置滑动条（初始值，最大值，回调函数）
    cv2.createTrackbar('minDist', 'Circle Tuner', 50, 200, nothing)
    cv2.createTrackbar('param1', 'Circle Tuner', 100, 255, nothing)
    cv2.createTrackbar('param2', 'Circle Tuner', 30, 100, nothing)
    cv2.createTrackbar('minR', 'Circle Tuner', 10, 200, nothing)
    cv2.createTrackbar('maxR', 'Circle Tuner', 200, 500, nothing)

    # 缓存上一次的参数，这是解决卡顿的【核心秘诀】
    prev_params = None
    res = img.copy()

    # 解决图片太小的问题：如果图片宽度或高度小于 400 像素，显示时自动放大 2 倍
    scale_factor = 1.0
    h, w = img.shape[:2]
    if max(h, w) < 400:
        scale_factor = 2.0 

    print(f"✅ 正在安全调优: {image_filename}")
    print("💡 提示：按 'ESC' 键退出并输出最终参数。")

    while True:
        # 获取当前参数，并设定【安全底线】，防止拉到 0 导致死机
        mDist = max(10, cv2.getTrackbarPos('minDist', 'Circle Tuner')) 
        p1 = max(10, cv2.getTrackbarPos('param1', 'Circle Tuner'))
        p2 = max(15, cv2.getTrackbarPos('param2', 'Circle Tuner')) # param2 低于 15 极易卡死，设为安全底线
        minR = max(1, cv2.getTrackbarPos('minR', 'Circle Tuner'))
        maxR = max(minR + 1, cv2.getTrackbarPos('maxR', 'Circle Tuner'))

        current_params = (mDist, p1, p2, minR, maxR)

        # 【重点】只有当你真正拖动了滑块（参数改变时），才执行耗时的霍夫检测
        if current_params != prev_params:
            res = img.copy()
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=mDist,
                                       param1=p1, param2=p2, 
                                       minRadius=minR, maxRadius=maxR)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(res, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 绿圈
                    cv2.circle(res, (i[0], i[1]), 2, (0, 0, 255), 3)    # 红心
            
            # 更新缓存
            prev_params = current_params

        # 处理显示比例
        if scale_factor != 1.0:
            display_img = cv2.resize(res, (0, 0), fx=scale_factor, fy=scale_factor)
        else:
            display_img = res

        cv2.imshow('Circle Tuner', display_img)
        
        # 增加 CPU 休息时间 (30ms)，让界面响应更丝滑
        if cv2.waitKey(30) & 0xFF == 27: 
            print(f"🎯 最终决定的参数 -> minDist:{mDist}, param1:{p1}, param2:{p2}, minR:{minR}, maxR:{maxR}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 在这里更改你要调优的图片名称
    run_circle_tuner("fig03.png")