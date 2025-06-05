# DBSCAN（Density-Based Spatial Clustering of Applications with Noise）
類型：分羣
優點：
不用指定羣數
可識別任意形狀羣體
自動標記離羣點（outliers）
學術引用高、常用於圖像、文本聚類、異常檢測
# 原理
DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是一種基於密度的非監督式分羣算法，適用於複雜形狀的數據集，並能自動辨識離羣點。
相較於 KMeans 必須指定羣數，DBSCAN 依據「密度」而非「中心點」來形成羣組，因此不需要預先設定類別數量，適合處理帶有噪聲或分佈不規則的數據。
```
X, _ = make_moons(n_samples=300, noise=0.1, random_state=0)
```
我們使用 make_moons() 生成一個非線性分佈的數據集，模擬兩個月牙形羣組，常用於測試無法被線性分割的分羣方法。
```
model = DBSCAN(eps=0.2, min_samples=5)
labels = model.fit_predict(X)
```
調用 DBSCAN 並設置兩個關鍵參數：
eps（ε）：半徑大小，表示每個點的鄰域範圍
min_samples：最少鄰居數，超過這個數量就被視爲「核心點」
演算法流程如下：
計算所有點的 ε 鄰域（距離小於 0.2 的點視爲鄰近）
將鄰域內有 ≥5 個點的標記爲「核心點」
將核心點相連的鄰域擴展形成羣組（density reachable）
不屬於任何羣組且鄰域不足的點 → 歸類爲噪聲點（-1）
# 结果解析
```
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Paired')
```
繪圖後可觀察：DBSCAN 成功識別出兩個曲線狀的羣組,同時自動標記孤立點爲離羣點（顏色不同）
