# localization 路徑飛行整合檢查

## 結論

這個專案適合驗證下列項目：

- 相機中心到下一個路徑點的三維連線與 XY 投影角度。
- 到新路徑點前先轉向，再將最新連線分解成機體前、右、上 PCMD。
- 真值位置與控制器估計位置／朝向分離。
- Sphinx 真值位置受限偏移下的 ANAFI 修正，以及每次控制輸出的 PCMD 紀錄。

這份模擬器的逐點懸停、轉向確認、再做 body-frame 三維平移，是指定的正式控制規格。
`定位演算法/flight_control` 現有的 continuous-polyline controller 不是這次要採用的
控制器。實機整合時應共用或搬移本專案的 `route.py` 控制邏輯，不能另外維護另一套
PCMD 公式。

EDM 定位已由操作者另外驗證為穩定可行，本模擬器不重跑 EDM 影像流程，而是使用其
觀察所得的 0.30 m 位置與 10°朝向誤差上限進行控制壓力測試。

## 目前壓力測試設定

測試設定只包含兩種誤差：

1. 每 20 秒透過 Sphinx `move_drone` 對真值位置施加一次隨機水平偏移，水平向量長度
   最多 0.40 m。
2. 每次控制更新的三維位置估計誤差向量長度最多 0.30 m，相機朝向估計誤差最多
   ±10°。

PCMD 沒有額外削弱、漏送或加入人工軸向誤差。真值三維距離只做測試評分，控制器
看不到目前真值；它用延遲 0.20 秒、時間相關且有 3% 更新遺失率的估計狀態做決策。
Sphinx 真值以 20 Hz 讀取，解析器只接受 `omniscient_anafi.worldPosition`，排除同一串流
中的垂直相機與 UE4 場景物件位置；控制決策與 PCMD 更新仍依需求維持 1 Hz。

位置 0.30 m 與朝向 10° 是操作者依 GlueMap/EDM 畫面觀察指定的壓力上限。專案內
四段真實 ANAFI 影片的 image-only 比較中，舊 deep 路徑相對完整
XFeat+LighterGlue 參考的最差影片位置差 P95 為 0.298 m、yaw 差 P95 為 2.807°。
這支持把 0.30 m 當成合理的壓力尺度；±10° 則刻意比該 P95 嚴格。該比較共用影像、
地圖、相機參數與 PnP，不是全站儀 ground truth，因此不能用來聲稱絕對定位精度。

真值驗收半徑維持 0.50 m。為預留 0.30 m 定位誤差與 0.20 秒延遲期間的移動量，控制器
使用 0.15 m 的估計到點半徑，估計速度不得超過 0.25 m/s，並要求連續三次更新符合；
接近點位時依剩餘距離減速，過速或關閉速度太高時送零 PCMD 煞停。
每個路徑點開始平移前，估計 XY 夾角還必須連續三次落在 3°內，且估計 yaw 角速度
不超過 5°/s；確認期間維持零 PCMD 懸停。

除 seeded random route 外，`worst-case` 提供五個不依賴隨機抽樣的固定案例：誤差永遠
朝向目標、誤差永遠背離目標、最後進點時 0.40 m 風偏移、非零 PCMD 後立刻定位遺失，
以及上述邊界的組合案例。

seed 42 的嚴格裕量完整測試結果：

- 10/10 個路徑點全部抵達，第一次降落成功；共記錄 810 筆每秒控制決策。
- 10 個真值到達誤差介於 0.0820 m 至 0.1782 m，全部小於 0.50 m；每點均有三筆確認，
  最大估計到達速度為 0.0487 m/s。
- 40 次風偏移的最大值為 0.3969 m；定位位置誤差最大 0.2612 m；yaw 誤差最大
  6.2818°；有效 pose 年齡均為 0.20 秒。
- 25 次定位遺失均先送零 PCMD 懸停，沒有連續超過終止門檻；沒有 altitude、route
  deviation 或低信心安全違規。

結果位於 `artifacts/runs/waypoint-route-seed42-full-safety-v6-true-radius-reserve-20260802`。
這份結果早於「三次低角速度轉向確認」修改，只保留為隨機誤差歷史基準。

新規格的 deterministic `combined-bounds` 結果位於
`artifacts/runs/waypoint-route-combined-bounds-20260802-attempt2`：10/10 點完成、第一次降落
成功，最大真值到點誤差 0.4405 m；0.30 m 朝目標固定偏差、10° yaw 固定偏差、非零
PCMD 後定位遺失及最後進點 0.40 m 偏移均有生效。401 筆控制決策包含 29 筆轉向穩定
等待與 9 筆第三次確認完成。

## 實機前必須一致的因素

### 1. 使用指定的同一份控制器

模擬和實機若各自維護 PCMD 公式，模擬通過不代表實機程式通過。指定規格是本模擬器
`src/anafi_pcmd_sim/route.py` 的逐點控制；實機程式應直接共用這份角度、分力、到點、
freshness、watchdog 與記錄邏輯，再由實機 adapter 提供 EDM pose 與 Olympe 介面。

### 2. 座標系、尺度與外參

`localization` 現在使用原始 GLOMAP frame，水平為 X/Z、重力向上為 -Y；本模擬器
使用 Gazebo ENU metre。進入控制器前必須明確完成：

- GLOMAP、Blender aligned、ENU、ANAFI NED 間的軸交換與正負號。
- 相機中心到機體中心的平移外參。
- 相機水平光軸到機體前向的 yaw offset。

正式控制不建立 GLOMAP map unit 到 metre 的尺度。map-space 只提供正規化方向，
到點與 route deviation 門檻保留在各場域的 map unit；0.30 m/s 水平上限只由新鮮的
airframe speed telemetry 保護。Sphinx 中的 metre 門檻只屬於模擬器，不得複製到場域
profile。

### 3. 相機朝向與機體朝向

ANAFI 的雲台機械軸只有 roll 與 pitch，yaw 是電子影像穩定，沒有可獨立控制的機械
yaw。因此水平 PCMD 的 body forward/right 應以機體 yaw 為準，再套用已校正的相機
yaw offset，不能直接使用定位器目前不符合水平面的 `pose.yaw`。

### 4. 控制頻率與延遲

本模擬器依需求每秒重新定位一次、注入 0.20 秒延遲，每個 PCMD 的有效期是 0.75 秒；
`localization` 的實機流程則以 20 Hz 發送 PCMD，且 0.5 秒以上的 pose 或影像會觸發
hover。仍需量測實際 GPU 推論時間、影像時間戳、傳輸延遲與 PCMD 生效時間，再用相同
cadence 重播。這些是系統時序，不需要增加新的隨機誤差種類。

### 5. PCMD 響應

`response` 會對 ±10% roll、pitch、yaw、gaz 做 Sphinx 階躍測試，輸出速度曲線、釋放後
煞停距離、停止時間與有效平均 yaw 響應。已驗證的 `pitch=10%` Sphinx 結果為峰值
0.717 m/s、命令期間平均 0.342 m/s、釋放後移動 0.429 m、1.875 s 後穩定低於
0.05 m/s。這不是實機校正值；真機仍需用相同命令與報告格式量測後才能取代。

### 6. 速度回授與到達判定

模擬器已用連續估計位置計算 ENU 速度，加入減速、過速／關閉速度煞停，並以 0.15 m
估計半徑預留誤差後，要求連續三筆低速估計。此設定讓 10 個真值到點全部進入 0.5 m。
實機整合時仍應優先使用飛控可提供的濾波速度，並在最後降落前另外確認位置速度與
yaw 已穩定。1 Hz 嚴格版本約需 13.5 分鐘，實機應提高定位更新率以降低時間與耗電。

### 7. 不可省略的安全層

模擬器目前已有 pose freshness、低信心 hover、重複定位失敗終止、PCMD watchdog、
30% 起飛電量門檻、高度與路徑偏離 geofence，以及 Ctrl+C 零 PCMD 後降落。`localization`
另有跳點拒絕、20 Hz 獨立 PCMD sender、GPS／geofence 與人工接管；共用控制器時仍須
保留較完整的一方，不能以模擬器的簡化 gate 取代。

目前 `SparseCloudCollisionMonitor` 明確沒有接入 production flight path。實機前仍需
獨立障礙物／淨空安全層；稀疏點雲只能做警告，不能當作唯一碰撞判斷。

## ANAFI 依據

[ANAFI White Paper v1.4](https://www.parrot.com/assets/s3fs-public/2020-07/white-paper_anafi-v1.4-en.pdf)
記載機體控制迴圈為 200 Hz、具有風修正、最高風阻 50 km/h，並說明雲台是 roll/pitch
兩個機械軸加 roll/pitch/yaw 電子穩定。這支持以機體 yaw 控制水平朝向並考慮風修正，
但不支持把電子影像 yaw 當成可獨立操控的機械朝向。本輪 0.40 m 風擾動是可重現、
有上限的真值位置壓力注入，不等同於白皮書中的物理風速模型。
