# Site profile

一份 profile 將單一場域的點雲、localization bundle、相機內參與選用的安全航線綁定在一起，避免將不同場域的資產混用。

請複製 `example_site_edm.json` 後填寫實際路徑。資產路徑可使用相對於 profile 所在目錄的路徑；提交設定時不要使用個人電腦的絕對路徑。

`route_json` 設為 `null` 時，該 profile 僅可用於離線 replay，不可用於真機飛行。
