#環境構築
import cv2
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import shutil
import glob
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
import pyrealsense2 as rs


#関数一覧

#labelファイルからBBボックスの情報を読み取る
def extract_boxes_from_file(file_path):
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            label = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            boxes.append((label, center_x, center_y, width, height))
    return boxes

#画像をBBボックスごとに切り取る
def crop_image(image, center_x, center_y, width, height):
    # 画像の高さと幅を取得
    h, w = image.shape[:2]

    # バウンディングボックスの左上と右下の座標を計算
    x_min = int((center_x - width / 2) * w)
    y_min = int((center_y - height / 2) * h)
    x_max = int((center_x + width / 2) * w)
    y_max = int((center_y + height / 2) * h)

    # 画像を切り取る
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image

#画像から赤のピクセルを抽出
def extract_red_pixels(image):
    # 赤色の範囲を定義 (BGRでの指定)
    lower_red = np.array([0, 0, 50])   #理想は150
    upper_red = np.array([255, 255, 255])

    # 赤色領域を抽出するマスクを作成
    mask = cv2.inRange(image, lower_red, upper_red)

    # 赤色領域のみを残して画像をマスク処理
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


#赤ピクセルから第一固有ベクトルを求める
def calculate_pca(image):
    # 赤色のピクセルを抽出
    red_pixels = extract_red_pixels(image)

    # 赤色のピクセルの座標を取得
    red_indices = np.where((red_pixels[:,:,0] != 0) & (red_pixels[:,:,1] != 0) & (red_pixels[:,:,2] != 0))
    red_pixels_coordinates = np.column_stack((red_indices[1], red_indices[0]))  # x, y座標の順に並ぶようにする

    # PCAを適用
    pca = PCA(n_components=2)
    pca.fit(red_pixels_coordinates)

    # 第一固有ベクトルを取得
    first_eigen_vector = pca.components_[0]

    return first_eigen_vector


#第一固有ベクトルと画像の枠との交点を算出
def plot_eigen_vectors(eigen_vectors, image_paths, output_file):
    with open(output_file, 'w') as f:
        for i, (eigen_vector, image_path) in enumerate(zip(eigen_vectors, image_paths)):
            # 画像を読み込む
            image = cv2.imread(image_path)

            # 画像の高さと幅を取得
            h, w = image.shape[:2]

            # 四辺の交点の座標を計算
            x_top = int(w / 2 - eigen_vector[0] * h / 2 / eigen_vector[1])
            y_top = 0
            x_bottom = int(w / 2 + eigen_vector[0] * h / 2 / eigen_vector[1])
            y_bottom = h

            # 交点の座標を出力
            f.write(f"Image {i + 1} の交点の座標:\n")
            f.write(f"上側の交点: ({x_top}, {y_top})\n")
            f.write(f"下側の交点: ({x_bottom}, {y_bottom})\n\n")

            # 画像上に交点をプロット
            '''確認用
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.scatter([x_top, x_bottom], [y_top, y_bottom], color='r', marker='x')
            plt.title(f"Image {i + 1}")
            plt.show()
            '''


#交点の上側のみを抽出
def extract_intersection_coordinates(file_path):
    coordinates = {}
    with open(file_path, 'r') as f:
        content = f.read()
        # "Image" の位置でテキストを分割
        sections = content.split("Image ")
        for section in sections[1:]:
            lines = section.strip().split("\n")
            # "Image X" から X を取得
            image_number = int(lines[0].split()[0])
            # 上側の交点の座標を抽出して辞書に追加
            Top_coordinates = lines[1].split(":")[1].strip()[1:-1]
            x, y = map(int, Top_coordinates.split(","))
            coordinates[f"koutenx{image_number}"] = x
            coordinates[f"kouteny{image_number}"] = y
    return coordinates


#特定の座標に最も近い、特定のラベル（この場合はラベル0）の検出ボックスを見つける
def find_nearest_label_zero_box(boxes, Top_x, Top_y, image):
    label_zero_boxes = [box for box in boxes if box[0] == 0]
    if not label_zero_boxes:
        return None
    nearest_box = None
    min_distance = float('inf')
    for label, center_x, center_y, width, height in label_zero_boxes:
        box_bottom_center_x = center_x * image.shape[1]
        box_bottom_center_y = (center_y + height / 2) * image.shape[0]
        distance = ((Top_x - box_bottom_center_x) ** 2 + (Top_y - box_bottom_center_y) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            nearest_box = (label, center_x, center_y, width, height)
    return nearest_box


#画像内の検出された物体の座標情報をテキストファイルに保存する

def save_coordinates_to_file(coordinates_data, output_file):
    with open(output_file, 'w') as file:
        for data in coordinates_data:
            file.write(f"Label: {data['label']}, Coordinates: ({data['x_min']}, {data['y_min']}) to ({data['x_max']}, {data['y_max']})\n")


#上側の交点のみをテキストファイルに保存
def extract_intersection_coordinates(file_path):
    coordinates = {}
    with open(file_path, 'r') as f:
        content = f.read()
        # "Image" の位置でテキストを分割
        sections = content.split("Image ")
        for section in sections[1:]:
            lines = section.strip().split("\n")
            # "Image X" から X を取得
            image_number = int(lines[0].split()[0])
            # 上側の交点の座標を抽出して辞書に追加
            Top_coordinates = lines[1].split(":")[1].strip()[1:-1]
            x, y = map(int, Top_coordinates.split(","))
            coordinates[f"koutenx{image_number}"] = x
            coordinates[f"kouteny{image_number}"] = y
    return coordinates

# ファイルから座標を読み取る関数
def read_coordinates_from_file(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            # 座標の抽出
            start_index = line.find("(") + 1
            end_index = line.find(")")
            coord_str = line[start_index:end_index]
            x, y = map(int, coord_str.split(','))
            coordinates.append((x, y))
    return coordinates

# 2つの座標を足す関数
def add_coordinates(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return x1 + x2, y1 + y2

#誤差を計算してその座標を特定する
def find_best_coordinate(peduncle_coordinates, midpoint_coordinates):
    min_mse = float('inf')
    best_coordinate = None

    
    for peduncle_coord, midpoint_coord in zip(peduncle_coordinates, midpoint_coordinates):
        combined_coord = (peduncle_coord[0] + midpoint_coord[0], peduncle_coord[1] + midpoint_coord[1])

        # 誤差を計算
        distances = np.abs(coefficients[0] * y - x + coefficients[1]) / np.sqrt(coefficients[0] ** 2 + 1)
        mse = np.mean(distances ** 2)
        if mse < min_mse:
            min_mse = mse
            best_coordinate = combined_coord
    return best_coordinate

#関数一覧終わり



#mainプログラム

#最初にフォルダを削除
# 削除したいフォルダのパス
folder_path = "./runs"

# フォルダが存在する場合は削除
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
    print("フォルダが削除されました。")
    
#最初にファイルを削除
# 削除したいファイルのリスト
file_paths = ["combined_coordinates_plot.png", "combined_coordinates_plot2.png", "coordinates_output.txt", "depth_image.png", "ichigo.jpg","ichigo.png", "intersection_coordinates.txt", "label_0_0_cropped.jpg", "label_0_1_cropped.jpg", "label_0_2_cropped.jpg", "label_0_3_cropped.jpg", "label_0_4_cropped.jpg", "label_0_5_cropped.jpg", "label_1_0_cropped.jpg", "label_1_1_cropped.jpg", "label_1_2_cropped.jpg", "label_1_3_cropped.jpg", "label_1_4_cropped.jpg", "label_1_5_cropped.jpg", "label_2_0_cropped.jpg", "label_2_1_cropped.jpg", "label_2_2_cropped.jpg", "label_2_3_cropped.jpg", "label_2_4_cropped.jpg", "label_2_5_cropped.jpg","midpoints.txt","peduncle_1_close.jpg","peduncle_2_close.jpg","Top_coordinates.txt", "result.txt"]
 
# ファイルが存在する場合は削除
for file_path in file_paths:
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} が削除されました。")
        


# ストリーム(Depth/Color)の設定
config = rs.config()
config.enable_stream(rs.stream.color, 640*2, 360*2, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640*2, 360*2, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)

# Alignオブジェクト生成
align_to = rs.stream.color
align = rs.align(align_to)

try:
    # フレーム待ち(Color & Depth)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not depth_frame or not color_frame:
        print("フレームが取得できませんでした。")
    else:
        # imageをnumpy arrayに
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # depth imageをカラーマップに変換
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)

       
        # 画像を保存
        cv2.imwrite("ichigo.jpg", color_image)
        cv2.imwrite("depth_image.png", depth_image)
        cv2.waitKey(0)

finally:
    # ストリーミング停止
    pipeline.stop()
    cv2.destroyAllWindows()

###############################################





'''

# 入力画像のファイルパス
input_image_path = "ichigo.png"

# 出力画像のファイルパス
output_image_path = "ichigo.jpg"

# 画像を読み込む
image = cv2.imread(input_image_path)

# 画像をJPEG形式で保存する
cv2.imwrite(output_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
'''
#入力は以下の通り
image_path = "ichigo.jpg"

#VIT_YOLOで苺、茎を検出
model = YOLO('best.pt')
results = model(image_path,save=True,save_txt=True,save_conf=True)

#####################################

#ここに検出をやり直すのを書く




#####################################



# バウンディングボックスファイルパス
boxes_file_path = os.path.join("runs","detect","predict","labels","ichigo.txt")

# 画像を読み込む
image = cv2.imread(image_path)

# バウンディングボックスの情報をファイルから読み込む
boxes = extract_boxes_from_file(boxes_file_path)

# 各バウンディングボックスごとに画像を切り取って保存
for i, box in enumerate(boxes):
    label, center_x, center_y, height, width = box
    cropped_image = crop_image(image, center_x, center_y, height, width)
    cv2.imwrite(f"label_{label}_{i}_cropped.jpg", cropped_image)

# 画像ファイルパスのパターン
image_pattern = "label_1_{}_cropped.jpg"

# 各画像の第一固有ベクトルを保存するリスト
eigen_vectors = []

# 画像のファイルパスを保存するリスト
image_paths = []

# 画像ごとにPCAを実行
for i in range(10):
    image_path = image_pattern.format(i)
    if not os.path.exists(image_path):
        continue  # ファイルが見つからない場合は次の画像へ
    if i == 10:
        break 
    
    # 画像を読み込む
    image = cv2.imread(image_path)
    # PCAを計算
    first_eigen_vector = calculate_pca(image)
    eigen_vectors.append(first_eigen_vector)
    image_paths.append(image_path)

# 四辺の交点をプロットして表示し、座標を出力
output_file = "intersection_coordinates.txt"
plot_eigen_vectors(eigen_vectors, image_paths, output_file)

# テスト用ファイルパス
file_path = "intersection_coordinates.txt"
# 交点の座標を抽出して辞書として取得
coordinates = extract_intersection_coordinates(file_path)
# 辞書内の座標を表示(確認用)
#print(coordinates)

#交点ファイルと出力ファイルとイメージパス
intersection_coordinates_file = "intersection_coordinates.txt"
output_file = "Top_coordinates.txt"
image_path = "ichigo.jpg"
#Top_coordinates.txtに上部の座標を保存
image = cv2.imread(image_path)
boxes = extract_boxes_from_file(boxes_file_path)
intersection_coordinates = extract_intersection_coordinates(intersection_coordinates_file)
with open(output_file, 'w') as f:
    for i, box in enumerate(boxes):
        label, center_x, center_y, width, height = box
        if label != 1:
            continue
        x_min = int((center_x - width / 2) * image.shape[1])
        y_min = int((center_y - height / 2) * image.shape[0])
        Top_x = intersection_coordinates.get(f"koutenx{i + 1}", 0)
        Top_y = intersection_coordinates.get(f"kouteny{i + 1}", 0)
        Top_x_global = x_min + Top_x
        Top_y_global = y_min + Top_y
        f.write(f"Box {i + 1} Top Coordinate: ({Top_x_global}, {Top_y_global})\n")

#上部の交点に最も近いBBボックスを特定し、その画像を保存
Top_coordinates_file = "Top_coordinates.txt"
output_file = "coordinates_output.txt"
image = cv2.imread(image_path)
boxes = extract_boxes_from_file(boxes_file_path)
with open(Top_coordinates_file, 'r') as file:
    Top_coordinates = [line.strip().split(": ")[1] for line in file]
coordinates_data = []
for i, Top_coordinate in enumerate(Top_coordinates, 1):
    Top_x, Top_y = map(int, Top_coordinate.strip("()").split(", "))
    nearest_box = find_nearest_label_zero_box(boxes, Top_x, Top_y, image)
    if nearest_box is not None:
        label, center_x, center_y, width, height = nearest_box
        x_min = int((center_x - width / 2) * image.shape[1])
        y_min = int((center_y - height / 2) * image.shape[0])
        x_max = int((center_x + width / 2) * image.shape[1])
        y_max = int((center_y + height / 2) * image.shape[0])
        #cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cropped_img = crop_image(image, center_x, center_y, width, height)
        cv2.imwrite(f"peduncle_{i}_close.jpg", cropped_img)
        coordinates_data.append({
            "label": label,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max
        })
save_coordinates_to_file(coordinates_data, output_file)
#特定したBBボックス座標確認用　coordinates_data


# 画像ファイルのパターン
file_pattern = 'peduncle_{}_close.jpg'

# 画像が存在するかチェックしてから処理を行う
for i in range(1, 10):
    file_name = file_pattern.format(i)
    
    # 画像が存在するかチェック
    if not os.path.exists(file_name):
        
        continue
    
    img = cv2.imread(file_name)

    # 画像の読み込みが成功した場合のみ処理を行う
    if img is not None:
        # BGRからHSVに変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 緑色の範囲を定義 (HSV形式で)
        lower_green = np.array([35, 100, 50])
        upper_green = np.array([70, 255, 255])

        # 緑色の範囲に基づいてマスクを作成
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # マスクを使って元の画像から緑色の部分を抽出
        green_pixels = cv2.bitwise_and(img, img, mask=mask)

        # 緑色のピクセルの座標を取得
        indices = np.where(green_pixels != 0)
        x = indices[1]  # x座標
        y = indices[0]  # y座標

        # 近似直線を計算、座標変換も行う
        coefficients = np.polyfit(y, x, 1)
        poly = np.poly1d(coefficients)


        # 近似直線の両端の座標を計算
        y_min, y_max = np.clip(np.min(y), 0, img.shape[0]-1), np.clip(np.max(y), 0, img.shape[0]-1)
        x_min, x_max = int(poly(y_min)), int(poly(y_max))

        # 近似直線の中点の座標を計算
        midpoint_x = (x_min + x_max) // 2
        midpoint_y = (y_min + y_max) // 2

        # ファイルに座標を書き込む
        with open('midpoints.txt', 'a') as file:
            file.write(f'File: {file_name}, Midpoint Coordinates: ({midpoint_x}, {midpoint_y})\n')
'''
        # 緑色の分布をプロット
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 元の画像を表示
        plt.scatter(x, y, color='blue', s=2)  # 緑色のピクセルをプロット
        plt.plot(poly(y), y, color='red')  # 近似直線をプロット
        plt.title(f'{file_name}')
        plt.show()
'''
# coordinates_output.txt から座標を読み取る
peduncle_coordinates = read_coordinates_from_file("coordinates_output.txt")

# midpoints.txt から座標を読み取る
midpoint_coordinates = read_coordinates_from_file("midpoints.txt")

# 画像ファイルのパターン
file_pattern = 'peduncle_{}_close.jpg'


# 最も誤差が小さい座標を特定する
best_coordinate = find_best_coordinate(peduncle_coordinates, midpoint_coordinates)
#z座標を特定する
best_x, best_y = best_coordinate
################################################################
# ピクセルの周囲の範囲（例：3x3の領域だったらneighborhood_size = 3）
neighborhood_size = 4
half_size = neighborhood_size // 2

# 周囲のピクセルの座標を保存するリスト
neighborhood_coordinates = []

# ピクセルの周囲の座標を取得する
for dx in range(-half_size, half_size + 1):
    for dy in range(-half_size, half_size + 1):
        # ピクセルの座標を計算
        x_pixel = best_x + dx
        y_pixel = best_y + dy
        
        # ピクセルの座標をリストに追加
        neighborhood_coordinates.append((x_pixel, y_pixel))

# ピクセルの周囲の座標に対応する深度データを取得して、平均値を計算
depth_data_sum = 0.0
for x_pixel, y_pixel in neighborhood_coordinates:
    # 深度データを取得
    depth_data = depth_frame.get_distance(x_pixel, y_pixel)
    print(depth_data)
    # 取得した深度データを合計に加算
    depth_data_sum += depth_data
    
    ################################################
    
    #0を抜き、もしすべて0なら最初からやり直す
    
    
    ##################################################
    
    

# 平均深度を計算
average_depth = depth_data_sum / len(neighborhood_coordinates)
print("平均深度:", average_depth)

#x,y座標をピクセル数からメートルに変換
#best_x2 = best_x*(average_depth/0.0913)*(1/(720**2+1280**2)**(1/2))
#best_y2 = best_y*(average_depth/0.0913)*(1/(720**2+1280**2)**(1/2))
best_x2 = (best_x-654.167)*average_depth/913.710
best_y2 = (best_y-374.424)*average_depth/912.038
print(best_x)
print(best_y)
best_coordinate2 = best_x2 , best_y2

#z座標を追加
best_coordinate = best_coordinate2 + (average_depth,)


#最終座標を出力
print(f"Best Coordinate(単位はメートル): {best_coordinate}")
print(f"茎の傾きの逆数: {-1/((y_max-y_min)/(x_max-x_min))}")
      
      
      
      ###############################################################################################################
# coordinates_output.txt から座標を読み取る
peduncle_coordinates = read_coordinates_from_file("coordinates_output.txt")

# midpoints.txt から座標を読み取る
midpoint_coordinates = read_coordinates_from_file("midpoints.txt")

# 画像の読み込み
img = plt.imread("depth_image.png")
# 画像をプロット
plt.imshow(img)

# 座標をプロット
for i in range(len(peduncle_coordinates)):
    peduncle_coord = peduncle_coordinates[i]
    midpoint_coord = midpoint_coordinates[i]

    # 2つの座標を足して合成座標を計算
    combined_coord = add_coordinates(peduncle_coord, midpoint_coord)

  

    # プロット
    plt.plot(*combined_coord, 'mo')  # 黒い点で座標をプロット
    # plt.text(*combined_coord, f'({combined_coord[0]},{combined_coord[1]})', fontsize=8, color='white')  # 座標を表示

# 図を保存
plt.savefig("combined_coordinates_plot2.png")

# 図を表示
plt.show()

# 画像の読み込み
img = plt.imread("ichigo.jpg")
# 画像をプロット
plt.imshow(img)

# 座標をプロット
for i in range(len(peduncle_coordinates)):
    peduncle_coord = peduncle_coordinates[i]
    midpoint_coord = midpoint_coordinates[i]

    # 2つの座標を足して合成座標を計算
    combined_coord = add_coordinates(peduncle_coord, midpoint_coord)

  

    # プロット
    plt.plot(*combined_coord, 'mo')  # 黒い点で座標をプロット
    # plt.text(*combined_coord, f'({combined_coord[0]},{combined_coord[1]})', fontsize=8, color='white')  # 座標を表示

# 図を保存
plt.savefig("combined_coordinates_plot.png")

# 図を表示
plt.show()
###############################################################################################################################

