# 控制介面程式

- `operator_interface/`：操作 UI、即時定位 worker 與 ANAFI 後端。
- `site_profiles/`：每個場域的一組原子化定位資產設定。
- `影片模擬串流/`：以預錄影片進行離線驗證的入口。
- `真機串流/`：真機操作入口，必須明確指定已驗證的 site profile。

先從 `site_profiles/example_site_edm.json` 建立自己的設定。真機啟動器不會預設選取任何場域。
