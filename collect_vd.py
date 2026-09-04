"""
VD 快照連續蒐集(collect_vd.py)—— 長時間掛著抓台北市 VD 資料
====================================================================
台北開放資料的 VD 每 5 分鐘更新一次,但**更新時刻不固定**(視上游批次而定)。
若固定每 300 秒抓一次,會發生兩種事:
  - 抓到跟上一份完全一樣的資料(重複檔,浪費硬碟)
  - 相位一漂就整份跳過(資料斷點)
本腳本改成「高頻探測、以資料自己的 ExchangeTime 去重」:
每 60 秒探測一次,只有 <ExchangeTime> 變動時才存檔 → 不漏、不重。

檔名用**資料時戳**而非下載時戳(VD_20260428_103103.xml.gz),
所以重跑不會產生重複檔,排序即為時序。

用法(在你電腦、可連台北開放資料的網路):
  python collect_vd.py                      # 一直跑,Ctrl-C 停
  python collect_vd.py --hours 24           # 抓滿 24 小時自動停
  python collect_vd.py --poll 60            # 探測間隔(秒),預設 60
  python collect_vd.py --live-only          # 只抓設備級(本場景不建議,會缺地真)
  python collect_vd.py --no-gzip            # 存未壓縮(相容舊檔,但佔 7 倍空間)
  python collect_vd.py --max-gb 5           # traffic_data/ 超過 5GB 就停

兩支來源都要抓,因為它們是不同東西:
  GetVDDATA.xml.gz → 設備級,鍵是 DeviceID,逐車道 Svolume(小客車)
  GetVD.xml.gz     → 路段級,鍵是 SectionId,逐路段 TotalVol + 座標
★ 和平東路的 vd_sumo_mapping.csv 用的是 **SectionId**(欄名雖叫 DeviceID),
  所以本場景的地真來自 **GetVD**;GetVDDATA 只用來算全市小客車比例 s_ratio。
  兩支缺一,validate_twin_fidelity 就算不出 GEH。

產出:
  traffic_data/GetVD_<資料時戳>.xml.gz   路段級:本場景的地真來源
  traffic_data/VD_<資料時戳>.xml.gz      設備級:算 s_ratio 用
  traffic_data/vd_index.csv              索引:時戳、檔名、對應到路網的流量
                                         → 之後挑尖峰/離峰不必再掃 500KB 檔案

注意:
  - traffic_data/ 目前是**被 git 追蹤的**。連抓一天就是 288 份,不要整包 commit,
    只把論文實際用到的那幾份加進版本控制。
  - 斷網不會中止蒐集:失敗會退避重試(60/120/240/480 秒上限),恢復後繼續。
"""
import argparse
import csv
import gzip
import io
import os
import re
import sys
import time
from datetime import datetime, timedelta

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "traffic_data")
INDEX_CSV = os.path.join(DATA_DIR, "vd_index.csv")
MAPPING_CSV = os.path.join(SCRIPT_DIR, "vd_sumo_mapping.csv")

STATIC_VD_URL = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz"
LIVE_VD_URL = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVDDATA.xml.gz"
UA = {"User-Agent": "NTUT_Research_Traffic_Collector/1.0"}

SOURCES = [("VD", LIVE_VD_URL), ("GetVD", STATIC_VD_URL)]
INDEX_COLS = ["exchange_time", "kind", "file", "units", "vol_all",
              "vol_mapped", "bytes"]

_EXCHANGE_RE = re.compile(r"<ExchangeTime>([^<]+)</ExchangeTime>")
_DEVICE_RE = re.compile(r"<DeviceID>([^<]+)</DeviceID>")
_SVOL_RE = re.compile(r"<Svolume>([^<]+)</Svolume>")
_SECID_RE = re.compile(r"<SectionId>([^<]+)</SectionId>")
_TOTVOL_RE = re.compile(r"<TotalVol>([^<]+)</TotalVol>")


# ── 抓取 ──────────────────────────────────────────────────────────
def fetch_gz(url, timeout=30):
    """下載 .gz 並解成文字。任何失敗都丟例外,由呼叫端決定退避。"""
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as f:
        return f.read().decode("utf-8", errors="replace")


# ── 解析(只用正則,不做完整 XML 剖析:500KB×288/天,省 CPU) ─────────
def exchange_time(text):
    """取資料自身的時戳 '2026/04/28T10:31:03' → '20260428_103103'。取不到回 None。"""
    m = _EXCHANGE_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip()
    for fmt in ("%Y/%m/%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y%m%d_%H%M%S")
        except ValueError:
            continue
    # 格式沒見過 → 退成安全檔名,至少不會漏存
    return re.sub(r"[^0-9]", "", raw)[:14] or None


def read_mapped_ids():
    """vd_sumo_mapping.csv 裡有對應到路網的識別碼集合(沒有檔案就回空集)。

    欄名是 DeviceID,但內容可能是 GetVDDATA 的 DeviceID **或** GetVD 的
    SectionId(和平東路場景是後者)。兩邊都拿它比對,對得上的那支就是地真來源。
    """
    if not os.path.exists(MAPPING_CSV):
        return set()
    out = set()
    with open(MAPPING_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            d = (row.get("DeviceID") or "").strip()
            if d and (row.get("SumoEdgeID") or "").strip():
                out.add(d)
    return out


def summarize_live(text, mapped):
    """設備級:(設備數, 全市 Svolume 總和, 對應到路網的 Svolume 總和)。

    逐 <VDDevice> 切段後累加,因此 vol_mapped 與 make_real_flow 的實際
    注入口徑一致(只算有對應到 SUMO edge 的設備)。
    """
    n_dev = 0
    all_sv = 0.0
    map_sv = 0.0
    for chunk in text.split("<VDDevice>")[1:]:
        m = _DEVICE_RE.search(chunk)
        if not m:
            continue
        n_dev += 1
        dev = m.group(1).strip()
        s = 0.0
        for v in _SVOL_RE.findall(chunk):
            try:
                s += float(v)
            except ValueError:
                pass
        all_sv += s
        if dev in mapped:
            map_sv += s
    return n_dev, all_sv, map_sv


def summarize_static(text, mapped):
    """路段級:(路段數, 全市 TotalVol 總和, 對應到路網的 TotalVol 總和)。"""
    n_sec = 0
    all_v = 0.0
    map_v = 0.0
    for chunk in text.split("<SectionData>")[1:]:
        m = _SECID_RE.search(chunk)
        t = _TOTVOL_RE.search(chunk)
        if not m or not t:
            continue
        n_sec += 1
        try:
            v = float(t.group(1))
        except ValueError:
            continue
        all_v += v
        if m.group(1).strip() in mapped:
            map_v += v
    return n_sec, all_v, map_v


# ── 存檔與索引 ────────────────────────────────────────────────────
def save_snapshot(text, prefix, stamp, use_gzip):
    """存成 traffic_data/<prefix>_<stamp>.xml[.gz];已存在則不覆寫,回 None。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    base = f"{prefix}_{stamp}.xml"
    plain = os.path.join(DATA_DIR, base)
    gzpath = plain + ".gz"
    if os.path.exists(plain) or os.path.exists(gzpath):
        return None
    if use_gzip:
        with gzip.open(gzpath, "wt", encoding="utf-8") as f:
            f.write(text)
        return gzpath
    with open(plain, "w", encoding="utf-8") as f:
        f.write(text)
    return plain


def append_index(row):
    new = not os.path.exists(INDEX_CSV)
    with open(INDEX_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def dir_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


# ── 主迴圈 ────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="連續蒐集台北市 VD 快照(以 ExchangeTime 去重)")
    p.add_argument("--poll", type=int, default=60,
                   help="探測間隔秒數(預設 60;資料每 5 分鐘更新,探測要比它密)")
    p.add_argument("--hours", type=float, default=None,
                   help="蒐集時數,不給則一直跑到 Ctrl-C")
    p.add_argument("--live-only", action="store_true",
                   help="只抓設備級 GetVDDATA,不抓路段級 GetVD")
    p.add_argument("--no-gzip", action="store_true",
                   help="存未壓縮 .xml(相容舊檔,但約佔 7 倍空間)")
    p.add_argument("--max-gb", type=float, default=None,
                   help="traffic_data/ 超過此大小(GB)就停止蒐集")
    args = p.parse_args()

    sources = SOURCES[:1] if args.live_only else SOURCES
    use_gzip = not args.no_gzip
    mapped = read_mapped_ids()
    deadline = (datetime.now() + timedelta(hours=args.hours)) if args.hours else None
    max_bytes = int(args.max_gb * 1024 ** 3) if args.max_gb else None

    print(f"蒐集來源:{', '.join(k for k, _ in sources)}")
    print(f"探測間隔:{args.poll}s   壓縮:{'gzip' if use_gzip else '否'}   "
          f"對應表識別碼:{len(mapped)} 個"
          f"{'(空 → vol_mapped 會是 0)' if not mapped else ''}")
    print(f"結束條件:{deadline.strftime('%Y-%m-%d %H:%M:%S') if deadline else 'Ctrl-C'}")
    print(f"存放:{DATA_DIR}\n")

    last_stamp = {k: None for k, _ in sources}
    saved = {k: 0 for k, _ in sources}
    backoff = args.poll
    polls = 0

    try:
        while True:
            if deadline and datetime.now() >= deadline:
                print("\n[結束] 已達設定時數。")
                break
            if max_bytes and dir_bytes(DATA_DIR) > max_bytes:
                print(f"\n[結束] traffic_data/ 已超過 {args.max_gb} GB。")
                break

            polls += 1
            failed = False
            for kind, url in sources:
                try:
                    text = fetch_gz(url)
                except Exception as e:                      # noqa: BLE001
                    print(f"[{datetime.now():%H:%M:%S}] {kind} 抓取失敗:{e}")
                    failed = True
                    continue

                stamp = exchange_time(text)
                if stamp is None:
                    print(f"[{datetime.now():%H:%M:%S}] {kind} 無 ExchangeTime,略過")
                    continue
                if stamp == last_stamp[kind]:
                    continue                                # 資料還沒更新
                last_stamp[kind] = stamp

                path = save_snapshot(text, kind, stamp, use_gzip)
                if path is None:
                    continue                                # 之前已抓過同一份

                if kind == "VD":
                    n_u, all_v, map_v = summarize_live(text, mapped)
                    unit = "設備"
                else:
                    n_u, all_v, map_v = summarize_static(text, mapped)
                    unit = "路段"
                append_index({
                    "exchange_time": stamp, "kind": kind,
                    "file": os.path.basename(path),
                    "units": n_u,
                    "vol_all": f"{all_v:.0f}",
                    "vol_mapped": f"{map_v:.0f}",
                    "bytes": os.path.getsize(path),
                })
                saved[kind] += 1
                extra = (f"  {unit} {n_u}  全市 {all_v:.0f}  "
                         f"路網內 {map_v:.0f}")
                print(f"[{datetime.now():%H:%M:%S}] ✓ {os.path.basename(path)}"
                      f"  ({os.path.getsize(path)/1024:.0f} KB){extra}")

            backoff = min(backoff * 2, 480) if failed else args.poll
            if failed:
                print(f"    → {backoff}s 後重試")
            time.sleep(backoff)
    except KeyboardInterrupt:
        print("\n[中斷] 收到 Ctrl-C。")

    total = sum(saved.values())
    print(f"\n探測 {polls} 次,新增 "
          + "、".join(f"{k} {v} 份" for k, v in saved.items())
          + f"(共 {total} 份)")
    print(f"traffic_data/ 目前 {dir_bytes(DATA_DIR)/1024**2:.0f} MB")
    if total:
        print(f"索引:{os.path.relpath(INDEX_CSV, SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
