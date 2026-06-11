"""
從原始(校正自真實數據)的轉向比例，產生「持續一段時間」的車流。
原始 flow 是「300 秒內 number 台車」，跑完路網就空了。
這裡換算成 vehsPerHour，讓車輛在整段模擬時間內持續進場。

調整方式：
- DURATION：模擬秒數（想讓 RL 跑多久就設多長）
- DENSITY ：車流密度倍率（1.0 = 與真實同，2.0 = 兩倍車）
"""

import os
# 自動切換到此腳本所在資料夾，確保輸出檔產生在正確位置
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DURATION = 3600      # 模擬秒數（1 小時）
DENSITY  = 1.0       # 密度倍率
OUT      = "real_traffic_continuous.rou.xml"

# (from, to, departLane, number_in_300s) ← 直接來自你的原始檔，比例完全保留
FLOWS = [
    ("238526531#0",  "506322846#0", "0",  7),
    ("238526531#0",  "506322846#0", "1", 13),
    ("238526531#0",  "506322846#0", "2", 23),
    ("506322846#0",  "506322851#0", "0", 12),
    ("506322846#0",  "288135626#0", "0",  6),
    ("506322846#0",  "506322851#0", "1", 16),
    ("506322846#0",  "288135626#0", "1",  8),
    ("506322846#0",  "506322851#0", "2", 11),
    ("506322846#0",  "288135626#0", "2",  5),
    ("1458697048#0", "238526531#0", "0",  1),
    ("1458697048#0", "288135626#0", "1",  9),
    ("1458697048#0", "506322851#0", "1",  3),
    ("1458697048#0", "238526531#0", "1",  3),
    ("1458697048#0", "238526531#0", "2",  1),
    ("1458697048#0", "288135626#0", "2",  7),
    ("1458697048#0", "506322851#0", "2",  3),
    ("1458697048#0", "238526531#0", "2",  2),
    ("182334379#0",  "288135628#0", "0",  6),
    ("182334379#0",  "238526531#0", "0",  2),
    ("182334379#0",  "506322851#0", "0",  2),
    ("182334379#0",  "288135628#0", "1", 12),
    ("182334379#0",  "238526531#0", "1",  5),
    ("182334379#0",  "506322851#0", "1",  4),
    ("182334379#0",  "506322851#0", "2",  1),
    ("182334379#0",  "288135628#0", "2", 10),
    ("182334379#0",  "238526531#0", "2",  4),
    ("182334379#0",  "506322851#0", "2",  3),
]

with open(OUT, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n\n')
    f.write('    <!-- 車輛類型：先用單一類型，server/client 角色在 Python 端指派 -->\n')
    f.write('    <vType id="passenger" vClass="passenger"/>\n\n')
    f.write(f'    <!-- 持續車流：begin=0 end={DURATION}，比例保留自真實數據 -->\n')
    total_vph = 0.0
    for i, (frm, to, lane, num) in enumerate(FLOWS):
        vph = num / 300.0 * 3600.0 * DENSITY     # 300秒num台 → 每小時車數
        total_vph += vph
        f.write(f'    <flow id="flow_{i}" type="passenger" '
                f'from="{frm}" to="{to}" begin="0" end="{DURATION}" '
                f'vehsPerHour="{vph:.1f}" departLane="{lane}" departSpeed="max"/>\n')
    f.write('\n</routes>\n')

print(f"已產生 {OUT}")
print(f"  時長 {DURATION}s、密度 x{DENSITY}、共 {len(FLOWS)} 條 flow")
print(f"  總進場率約 {total_vph:.0f} 台/小時 → 整段約 {total_vph*DURATION/3600:.0f} 台車")
