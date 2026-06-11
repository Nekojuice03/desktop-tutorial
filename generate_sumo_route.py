#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_sumo_route.py

即時從網頁抓取目前交通流量，並將流量動態轉換為 SUMO flow route。

輸出文件: real_traffic.rou.xml
時間區間: begin=0, end=300
"""

import argparse
import csv
import os
import subprocess
import sys
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

try:
    import traci
except ImportError:
    traci = None

LIVE_VD_URL = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVDDATA.xml.gz"
FLOW_CSV_FILE = "vd_sumo_flow.csv"
OUTPUT_ROUTE_FILE = "real_traffic.rou.xml"
MAPPING_FILE = "vd_sumo_mapping.csv"
TURNING_FILE = "turning_ratio.csv"


import gzip
import random

JITTER_RATIO = 0.10  # ±10% 隨機偏差

def apply_jitter(volume):
    """對車流量加上 ±JITTER_RATIO 的隨機偏差"""
    if volume <= 0:
        return 0
    return max(1, int(round(volume * random.uniform(1 - JITTER_RATIO, 1 + JITTER_RATIO))))



def fetch_live_traffic_xml(url=LIVE_VD_URL, timeout=15):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'traffic_data')
        os.makedirs(data_dir, exist_ok=True)

        response = requests.get(url, headers={'User-Agent': 'NTUT_Research_Traffic_Collector/1.0'}, timeout=timeout)
        if response.status_code != 200:
            print(f"無法取得即時交通資料，狀態碼: {response.status_code}")
            return None

        data = response.content
        if not data:
            print("抓取到的即時交通資料為空。")
            return None

        if data[:2] == b'\x1f\x8b':
            try:
                data = gzip.decompress(data)
            except Exception as e:
                print(f"解壓 gzip 資料失敗: {e}")
                return None

        try:
            xml_text = data.decode('utf-8')
        except UnicodeDecodeError:
            try:
                xml_text = data.decode('utf-8-sig')
            except UnicodeDecodeError as e:
                print(f"解碼 XML 資料失敗: {e}")
                return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(data_dir, f"VD_{timestamp}.xml")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(xml_text)
        print(f"[{datetime.now()}] 成功下載並保存: {file_path}")

        return xml_text
    except Exception as e:
        print(f"抓取即時交通資料失敗: {e}")
    return None


def parse_traffic_xml_string(xml_text):
    try:
        if xml_text is None:
            return []

        def to_int(value, default=0):
            try:
                return int(float(value))
            except Exception:
                return default

        def to_float(value, default=0.0):
            try:
                return float(value)
            except Exception:
                return default

        xml_text = xml_text.lstrip('\ufeff \t\r\n')
        if not xml_text.startswith('<?xml'):
            xml_text = '<?xml version="1.0" encoding="utf-8"?>\n' + xml_text

        root = ET.fromstring(xml_text)
        sections = []

        if root.tag.endswith('VDInfoSet'):
            for device in root.findall('.//VDDevice'):
                device_id = device.findtext('DeviceID', '').strip()
                for lane in device.findall('LaneData'):
                    sections.append({
                        'DeviceID': device_id,
                        'LaneNO': to_int(lane.findtext('LaneNO', '0')),
                        'Volume': to_int(lane.findtext('Volume', '0')),
                        'AvgSpeed': to_float(lane.findtext('AvgSpeed', '0')),
                        'AvgOccupancy': to_float(lane.findtext('AvgOccupancy', '0')),
                        'Svolume': to_int(lane.findtext('Svolume', '0')),
                        'Mvolume': to_int(lane.findtext('Mvolume', '0')),
                        'Lvolume': to_int(lane.findtext('Lvolume', '0')),
                    })
            return sections

        ns = {'vd': 'http://traffic.transportdata.tw/schema/'}
        if '<vd:ExchangeData' not in xml_text:
            xml_text = xml_text.replace('<SectionDataSet>', '<vd:SectionDataSet>') if '<SectionDataSet>' in xml_text else xml_text
            xml_text = '<vd:ExchangeData xmlns:vd="http://traffic.transportdata.tw/schema/">\n' + xml_text + '\n</vd:ExchangeData>'
            root = ET.fromstring(xml_text)

        for section in root.findall('.//vd:SectionData', ns):
            sections.append({
                'DeviceID': section.find('vd:SectionId', ns).text if section.find('vd:SectionId', ns) is not None else '',
                'LaneNO': 1,
                'Volume': to_int(section.findtext('vd:TotalVol', default='0', namespaces=ns)),
                'AvgSpeed': to_float(section.findtext('vd:AvgSpd', default='0', namespaces=ns)),
                'AvgOccupancy': to_float(section.findtext('vd:AvgOcc', default='0', namespaces=ns)),
                'Svolume': 0,
                'Mvolume': 0,
                'Lvolume': 0,
            })
        return sections
    except Exception as e:
        print(f"解析即時 XML 資料失敗: {e}")
        return []


def load_device_lane_mapping(mapping_file=MAPPING_FILE):
    """讀取 DeviceID + LaneNO 到 SumoEdgeID + departLane 的映射"""
    mapping = {}
    try:
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    device_id = row.get('DeviceID', '').strip()
                    lane_no = row.get('LaneNO', '').strip()
                    sumo_edge = row.get('SumoEdgeID', '').strip()
                    depart_lane = row.get('departLane', row.get('DepartLane', '')).strip()
                    if not device_id or not sumo_edge:
                        continue
                    if lane_no.isdigit():
                        key = (device_id, int(lane_no))
                    else:
                        key = (device_id, None)
                    mapping.setdefault(key, []).append({
                        'SumoEdgeID': sumo_edge,
                        'departLane': depart_lane or 'random'
                    })
        else:
            print(f"警告: 找不到映射文件 {mapping_file}")
    except Exception as e:
        print(f"載入映射文件時發生錯誤: {e}")
    return mapping


def load_turning_ratios(turning_file=TURNING_FILE):
    """讀取轉向比例設定"""
    ratios = {}
    try:
        if os.path.exists(turning_file):
            with open(turning_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    from_edge = row.get('from_edge', '').strip()
                    to_edge = row.get('to_edge', '').strip()
                    ratio_text = row.get('ratio', '').strip()
                    if not from_edge or not to_edge or not ratio_text:
                        continue
                    try:
                        ratio = float(ratio_text)
                    except ValueError:
                        continue
                    if ratio <= 0:
                        continue
                    ratios.setdefault(from_edge, []).append((to_edge, ratio))
            for from_edge, targets in ratios.items():
                total = sum(r for _, r in targets)
                if total > 0:
                    ratios[from_edge] = [(to_edge, r / total) for to_edge, r in targets]
            print(f"已載入 turning_ratio.csv，包含 {len(ratios)} 個起始 edge 的轉向比例")
        else:
            print("未找到 turning_ratio.csv，將使用映射中其他 edge 作為預設目的地進行分流")
    except Exception as e:
        print(f"載入 turning_ratio.csv 時發生錯誤: {e}")
    return ratios


def ensure_traci_available():
    if traci is None:
        raise RuntimeError(
            "TraCI 模組未安裝。請安裝 SUMO 並確保 Python 可匯入 traci 模組。"
        )


def launch_sumo_process(sumo_binary, sumo_cfg, port):
    if not sumo_cfg:
        raise ValueError("若要啟動 SUMO，請指定 --sumo-cfg SUMO 配置文件。")
    if not sumo_binary:
        sumo_binary = "sumo-gui"
    cmd = [sumo_binary, "--remote-port", str(port), "-c", sumo_cfg]
    print("啟動 SUMO:", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def connect_traci(port=8813, sumo_binary=None, sumo_cfg=None):
    ensure_traci_available()
    sumo_proc = None
    if sumo_cfg:
        if not sumo_binary:
            sumo_binary = "sumo-gui"
        sumo_proc = launch_sumo_process(sumo_binary, sumo_cfg, port)
        time.sleep(2)
    try:
        traci.connect(port=port)
        target = "SUMO GUI" if sumo_cfg and sumo_binary and "gui" in sumo_binary.lower() else "SUMO"
        print(f"已連線到 {target} 的 TraCI 伺服器 (port={port})")
    except Exception as e:
        if sumo_proc:
            sumo_proc.kill()
        raise RuntimeError(f"無法連線到 TraCI: {e}")
    return sumo_proc


def create_traci_route_id(from_edge, to_edge):
    return f"route_{sanitize_id(from_edge)}_to_{sanitize_id(to_edge)}"


def get_available_vehicle_types():
    try:
        return set(traci.vehicletype.getIDList())
    except Exception:
        return set()


def get_or_create_traci_route(from_edge, to_edge, route_cache):
    if not from_edge or not to_edge:
        return None
    route_id = create_traci_route_id(from_edge, to_edge)
    if route_id in route_cache:
        return route_id
    try:
        existing_routes = set(traci.route.getIDList())
        if route_id not in existing_routes:
            route_result = traci.simulation.findRoute(from_edge, to_edge)
            edges = getattr(route_result, 'edges', None)
            if not edges:
                print(f"TraCI 找不到路線: {from_edge} -> {to_edge}")
                return None
            traci.route.add(route_id, edges)
        route_cache.add(route_id)
        return route_id
    except Exception as e:
        print(f"建立 TraCI route 失敗 {from_edge}->{to_edge}: {e}")
        return None


def build_traci_flow_plan(traffic_data, device_mapping, turning_ratios):
    mapping_edges = sorted({info['SumoEdgeID'] for values in device_mapping.values() for info in values if info['SumoEdgeID']})
    flow_plan = []
    for data in traffic_data:
        mapping_info = get_device_lane_mapping(data['DeviceID'], data['LaneNO'], device_mapping)
        from_edge = mapping_info['SumoEdgeID'] if mapping_info and mapping_info.get('SumoEdgeID') else data.get('SumoEdgeID', '')
        depart_lane = mapping_info['departLane'] if mapping_info else 'random'
        if not from_edge:
            continue

        # 只保留小客車（Svolume），加入 ±10% 隨機偏差
        volume = apply_jitter(data.get('Svolume', 0))
        if volume <= 0:
            continue
        vtype = 'passenger'

        if from_edge in turning_ratios:
            targets = turning_ratios[from_edge]
        elif mapping_edges:
            equal_ratio = 1.0 / max(len(mapping_edges) - (1 if from_edge in mapping_edges else 0), 1)
            targets = [(to_edge, equal_ratio) for to_edge in mapping_edges if to_edge != from_edge]
        else:
            targets = [(from_edge, 1.0)]

        if not targets:
            targets = [(from_edge, 1.0)]

        remaining = volume
        for idx, (to_edge, ratio) in enumerate(targets):
            count = int(volume * ratio) if idx < len(targets) - 1 else remaining
            remaining -= count
            if count <= 0:
                continue
            flow_plan.append({
                'from_edge': from_edge,
                'to_edge': to_edge,
                'count': count,
                'type': vtype,
                'departLane': depart_lane
            })
    return flow_plan


def inject_traci_vehicles(flow_plan, route_cache, available_vtypes, interval, current_sim_time):
    injected_count = 0
    for flow_index, flow in enumerate(flow_plan):
        route_id = get_or_create_traci_route(flow['from_edge'], flow['to_edge'], route_cache)
        if not route_id:
            continue

        type_id = flow['type'] if flow['type'] in available_vtypes else None
        if type_id is None and available_vtypes:
            type_id = next(iter(available_vtypes))

        for i in range(flow['count']):
            depart_offset = int((i * interval) / max(flow['count'], 1))
            depart_time = int(current_sim_time + 1 + depart_offset)
            veh_id = f"veh_{sanitize_id(flow['from_edge'])}_{sanitize_id(flow['to_edge'])}_{flow['type']}_{flow_index}_{i}_{depart_time}"
            kwargs = {
                'routeID': route_id,
                'depart': depart_time,
                'departLane': 'best',
                'departSpeed': 'max'
            }
            if type_id:
                kwargs['typeID'] = type_id
            try:
                traci.vehicle.add(veh_id, **kwargs)
                injected_count += 1
            except Exception as e:
                print(f"無法注入車輛 {veh_id}: {e}")
    return injected_count


def run_traci_live_update(port=8813, sumo_binary=None, sumo_cfg=None, interval=60):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device_mapping = load_device_lane_mapping(os.path.join(script_dir, MAPPING_FILE))
    turning_ratios = load_turning_ratios(os.path.join(script_dir, TURNING_FILE))
    sumo_proc = connect_traci(port=port, sumo_binary=sumo_binary, sumo_cfg=sumo_cfg)
    route_cache = set()
    available_vtypes = get_available_vehicle_types()
    if not available_vtypes:
        print("警告: 未偵測到 SUMO 已定義車種，將使用 SUMO 預設車種。")

    print(f"開始 TraCI 即時更新，抓取 URL: {LIVE_VD_URL}")
    print(f"更新間隔: {interval} 秒。按 Ctrl+C 停止。\n")

    try:
        traci.simulationStep()  # 確保模擬開始
        next_update = int(traci.simulation.getTime())
        print(f"模擬開始，當前時間: {next_update}")
        while True:
            sim_time = int(traci.simulation.getTime())
            if sim_time >= next_update:
                xml_text = fetch_live_traffic_xml()
                if xml_text:
                    traffic_data = parse_traffic_xml_string(xml_text)
                    if traffic_data:
                        estimated = normalize_vehicle_type_counts(traffic_data)
                        if estimated > 0:
                            print(f"已估計 {estimated} 筆資料車種分布")
                        save_flow_csv(traffic_data, device_mapping, os.path.join(script_dir, FLOW_CSV_FILE))
                        generate_sumo_route_from_data(traffic_data, device_mapping, turning_ratios, os.path.join(script_dir, OUTPUT_ROUTE_FILE))
                        flow_plan = build_traci_flow_plan(traffic_data, device_mapping, turning_ratios)
                        injected = inject_traci_vehicles(flow_plan, route_cache, available_vtypes, interval, sim_time)
                        print(f"已透過 TraCI 注入 {injected} 台車輛。")
                    else:
                        print("解析後沒有交通資料，跳過此次更新。")
                else:
                    print("本次更新未取得即時交通資料。")
                next_update = sim_time + interval
            traci.simulationStep()
    except KeyboardInterrupt:
        print("\nTraCI 即時更新已停止。")
    except Exception as e:
        print(f"TraCI 執行時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            traci.close()
        except Exception:
            pass
        if sumo_proc:
            try:
                sumo_proc.terminate()
                sumo_proc.wait(timeout=5)
                print("已終止 SUMO 進程。")
            except Exception as e:
                print(f"終止 SUMO 進程時發生錯誤: {e}")
                try:
                    sumo_proc.kill()
                except Exception:
                    pass


def get_device_lane_mapping(device_id, lane_no, mapping):
    key = (device_id, lane_no)
    if key in mapping:
        values = mapping[key]
        return values[0] if values else None
    key = (device_id, None)
    values = mapping.get(key)
    return values[0] if values else None


def sanitize_id(value):
    return ''.join(ch if ch.isalnum() or ch in '_-' else '_' for ch in value)


def save_flow_csv(traffic_data, device_mapping, output_file=FLOW_CSV_FILE):
    data_rows = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for data in traffic_data:
        mapping_info = get_device_lane_mapping(data['DeviceID'], data['LaneNO'], device_mapping)
        sumo_edge = mapping_info['SumoEdgeID'] if mapping_info and mapping_info.get('SumoEdgeID') else data.get('SumoEdgeID', '')
        data_rows.append({
            'time': current_time,
            'DeviceID': data['DeviceID'],
            'SumoEdgeID': sumo_edge,
            'LaneNO': data['LaneNO'],
            'Volume': data['Volume'],
            'Svolume': data['Svolume'],
            'Mvolume': data['Mvolume'],
            'Lvolume': data['Lvolume']
        })
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data_rows[0].keys())
        writer.writeheader()
        writer.writerows(data_rows)
    print(f"已更新 {output_file}，包含 {len(data_rows)} 條記錄")


def generate_sumo_route_from_data(traffic_data, device_mapping, turning_ratios, output_file=OUTPUT_ROUTE_FILE):
    route_ids = {}
    if not turning_ratios:
        unique_edges = []
        for data in traffic_data:
            mapping_info = get_device_lane_mapping(data['DeviceID'], data['LaneNO'], device_mapping)
            edge = mapping_info['SumoEdgeID'] if mapping_info and mapping_info.get('SumoEdgeID') else data.get('SumoEdgeID', '')
            if edge and edge not in unique_edges:
                unique_edges.append(edge)
        for edge in unique_edges:
            route_ids[(edge,)] = f"route_{sanitize_id(edge)}"

    xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n\n'
    xml_content += '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n\n'
    xml_content += '    <!-- 車輛類型定義（僅小客車）-->\n'
    xml_content += '    <vType id="passenger" vClass="passenger"/>\n\n'

    if not turning_ratios:
        xml_content += '    <!-- 轉向比例缺失，使用映射中其他 edge 作為預設目的地進行測試分流 -->\n'
    else:
        xml_content += '    <!-- 轉向比例路線，使用 from/to 產生多條 flow -->\n'

    flow_count = 0
    for data in traffic_data:
        mapping_info = get_device_lane_mapping(data['DeviceID'], data['LaneNO'], device_mapping)
        from_edge = mapping_info['SumoEdgeID'] if mapping_info and mapping_info.get('SumoEdgeID') else data.get('SumoEdgeID', '')
        depart_lane = mapping_info['departLane'] if mapping_info else 'random'
        if not from_edge:
            continue

        # 只保留小客車（Svolume），加入 ±10% 隨機偏差
        volume = apply_jitter(data.get('Svolume', 0))
        if volume <= 0:
            continue
        vtype = 'passenger'

        if from_edge in turning_ratios:
            targets = turning_ratios[from_edge]
        else:
            mapping_edges = sorted({info['SumoEdgeID'] for values in device_mapping.values() for info in values if info['SumoEdgeID'] and info['SumoEdgeID'] != from_edge})
            if mapping_edges:
                equal_ratio = 1.0 / len(mapping_edges)
                targets = [(to_edge, equal_ratio) for to_edge in mapping_edges]
            else:
                targets = []

        if targets:
            remaining = volume
            for idx, (to_edge, ratio) in enumerate(targets):
                count = int(volume * ratio) if idx < len(targets) - 1 else remaining
                remaining -= count
                if count <= 0:
                    continue
                flow_id = f"flow_{flow_count}"
                xml_content += f'    <flow id="{flow_id}" type="{vtype}" from="{from_edge}" to="{to_edge}" begin="0" end="300" number="{count}" departLane="{depart_lane}" departSpeed="max"/>\n'
                flow_count += 1
        else:
            route_id = route_ids.get((from_edge,))
            if not route_id:
                route_id = f"route_{sanitize_id(from_edge)}"
                route_ids[(from_edge,)] = route_id
            flow_id = f"flow_{flow_count}"
            xml_content += f'    <flow id="{flow_id}" type="{vtype}" route="{route_id}" begin="0" end="300" number="{volume}" departLane="{depart_lane}" departSpeed="max"/>\n'
            flow_count += 1

    xml_content += '\n</routes>\n'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    print(f"已更新 SUMO route 文件: {output_file}，共 {flow_count} 個 flow")
    return flow_count


def normalize_vehicle_type_counts(traffic_data):
    estimated_count = 0
    for data in traffic_data:
        if data['Svolume'] == 0 and data['Mvolume'] == 0 and data['Lvolume'] == 0 and data['Volume'] > 0:
            total_volume = data['Volume']
            data['Svolume'] = int(total_volume * 0.25)
            data['Mvolume'] = int(total_volume * 0.65)
            data['Lvolume'] = int(total_volume * 0.10)
            estimated_total = data['Svolume'] + data['Mvolume'] + data['Lvolume']
            if estimated_total != total_volume:
                data['Mvolume'] += (total_volume - estimated_total)
            estimated_count += 1
    return estimated_count


def run_live_update(interval=60):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device_mapping = load_device_lane_mapping(os.path.join(script_dir, MAPPING_FILE))
    turning_ratios = load_turning_ratios(os.path.join(script_dir, TURNING_FILE))

    print(f"開始即時更新，抓取 URL: {LIVE_VD_URL}")
    print(f"更新間隔: {interval} 秒。按 Ctrl+C 停止。\n")

    while True:
        xml_text = fetch_live_traffic_xml()
        if xml_text:
            traffic_data = parse_traffic_xml_string(xml_text)
            if traffic_data:
                estimated = normalize_vehicle_type_counts(traffic_data)
                if estimated > 0:
                    print(f"已估計 {estimated} 筆資料車種分布")
                save_flow_csv(traffic_data, device_mapping, os.path.join(script_dir, FLOW_CSV_FILE))
                generate_sumo_route_from_data(traffic_data, device_mapping, turning_ratios, os.path.join(script_dir, OUTPUT_ROUTE_FILE))
            else:
                print("解析後沒有交通資料，跳過此次更新。")
        else:
            print("本次更新未取得即時交通資料。")

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taipei real-time traffic to SUMO route / TraCI updater")
    parser.add_argument('--interval', type=int, default=60, help='更新間隔秒數')
    parser.add_argument('--traci', action='store_true', help='使用 TraCI 動態注入車輛')
    parser.add_argument('--sumo-binary', default=None, help='SUMO 二進位檔，可為 sumo 或 sumo-gui；若使用 --sumo-cfg，預設啟動 sumo-gui')
    parser.add_argument('--sumo-gui', action='store_true', help='強制啟動 sumo-gui（若使用 --traci 且指定 --sumo-cfg）')
    parser.add_argument('--sumo-cfg', help='SUMO config 檔案，用於啟動 SUMO')
    parser.add_argument('--port', type=int, default=8813, help='TraCI 連線埠')
    args = parser.parse_args()

    try:
        if args.traci:
            if args.sumo_gui:
                args.sumo_binary = 'sumo-gui'
            elif args.sumo_cfg and not args.sumo_binary:
                args.sumo_binary = 'sumo-gui'

            run_traci_live_update(port=args.port, sumo_binary=args.sumo_binary, sumo_cfg=args.sumo_cfg, interval=args.interval)
        else:
            run_live_update(interval=args.interval)
    except KeyboardInterrupt:
        print("\n即時更新已停止。")
    except Exception as e:
        print(f"執行時發生錯誤: {e}")
        sys.exit(1)
