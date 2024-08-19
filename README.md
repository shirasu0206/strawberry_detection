# strawberry_detection

苺の検出に使う

以下、アルゴリズム

1. 画像からvit_yoloを使って苺、及び茎の座標を検出する
![ichigo](https://github.com/user-attachments/assets/ef1c207e-e593-4fc4-bab2-76078088bdac)
3. 苺のバウンディングボックス(以下BB)内の赤部分の点群からPCAを用いて苺の傾きを求める
4. 苺の傾きとBBの上辺の交点に最も近い茎のBBを検出する
5. 求めた茎のBB内の緑の点群から最小二乗法を用いて茎の近似直線を求める
6. 茎の近似直線の中点を切断位置とし、その点の奥行きを求める
![combined_coordinates_plot](https://github.com/user-attachments/assets/96d67033-7b3e-4111-8cf3-aab7dd7e925d)
8. 最終的な出力は切断位置の座標(x,y,z)と茎の傾き

mycobot280piを使ったdemo動画(hand無し)

https://github.com/user-attachments/assets/81b84b51-e3b0-4197-a6f2-94119999d076

