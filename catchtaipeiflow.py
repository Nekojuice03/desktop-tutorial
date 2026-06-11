import requests
import gzip
import io
import os
from datetime import datetime
import xml.etree.ElementTree as ET
import csv

def load_device_mapping(mapping_file='vd_sumo_mapping.csv'):
    """讀取DeviceID到SumoEdgeID的映射"""
    mapping = {}
    try:
        if not os.path.exists(mapping_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mapping_file = os.path.join(script_dir, mapping_file)

        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8-sig') as f:  # 使用 utf-8-sig 來處理 BOM
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    if 'DeviceID' in row and 'SumoEdgeID' in row:
                        device_id = row['DeviceID'].strip()
                        sumo_id = row['SumoEdgeID'].strip()
                        if device_id and sumo_id:
                            mapping[device_id] = sumo_id
                            count += 1
                print(f"已載入 {count} 個設備映射")
        else:
            print(f"警告: 找不到映射文件 {mapping_file} (當前目錄: {os.getcwd()})")
    except Exception as e:
        print(f"載入映射文件時發生錯誤: {e}")
    return mapping

def fetch_and_save_xml():
    url = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVDDATA.xml.gz"
    headers = {'User-Agent': 'NTUT_Research_Traffic_Collector/1.0'}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "traffic_data")
    os.makedirs(data_dir, exist_ok=True)

    try:
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 200:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(data_dir, f"VD_{timestamp}.xml")
            data = response.content

            if data[:2] == b'\x1f\x8b':
                try:
                    data = gzip.decompress(data)
                except Exception as e:
                    print(f"解壓 gzip 資料失敗: {e}")
                    return None

            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError:
                text = data.decode('utf-8-sig', errors='ignore')

            with open(file_path, "w", encoding="utf-8") as out:
                out.write(text)

            print(f"[{datetime.now()}] 成功下載並存檔: {file_path}")
            return file_path

        else:
            print(f"失敗，狀態碼: {response.status_code}")
            return None

    except Exception as e:
        print(f"發生錯誤: {e}")
        return None

def parse_traffic_data(xml_file):
    """解析 XML 文件並提取交通數據"""
    try:
        # 首先檢查並修復 XML 文件
        with open(xml_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 如果文件沒有正確的 XML 聲明，修復它
        if not content.startswith('<?xml'):
            # 總是修復XML結構，因為下載的文件缺少XML聲明和根元素
            prefix = '<?xml version="1.0" encoding="utf-8"?>\n<vd:ExchangeData xmlns:vd="http://traffic.transportdata.tw/schema/">\n<vd:SectionDataSet>\n'
            suffix = '\n</vd:SectionDataSet>\n</vd:ExchangeData>'
            fixed_content = prefix + content + suffix

            # 寫入修復的文件
            fixed_file = xml_file.replace('.xml', '_fixed.xml')
            with open(fixed_file, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            xml_file = fixed_file
            print(f"已修復 XML 文件: {fixed_file}")

        tree = ET.parse(xml_file)
        root = tree.getroot()

        sections = []
        if root.tag.endswith('VDInfoSet'):
            for device in root.findall('.//VDDevice'):
                device_id = device.findtext('DeviceID', '').strip()
                for lane in device.findall('LaneData'):
                    sections.append({
                        'DeviceID': device_id,
                        'LaneNO': int(float(lane.findtext('LaneNO', '0'))),
                        'Volume': int(float(lane.findtext('Volume', '0'))),
                        'AvgSpeed': float(lane.findtext('AvgSpeed', '0')),
                        'AvgOccupancy': float(lane.findtext('AvgOccupancy', '0')),
                        'Svolume': int(float(lane.findtext('Svolume', '0'))),
                        'Mvolume': int(float(lane.findtext('Mvolume', '0'))),
                        'Lvolume': int(float(lane.findtext('Lvolume', '0'))),
                    })
            return sections

        # 找到所有 SectionData 元素
        # 注意：XML 使用命名空間 vd:
        ns = {'vd': 'http://traffic.transportdata.tw/schema/'}
        for section in root.findall('.//vd:SectionData', ns):
            data = {
                'DeviceID': section.find('vd:SectionId', ns).text if section.find('vd:SectionId', ns) is not None else '',
                'LaneNO': 1,  # 默認車道號，因為XML中沒有此字段
                'Volume': int(float(section.find('vd:TotalVol', ns).text)) if section.find('vd:TotalVol', ns) is not None and section.find('vd:TotalVol', ns).text else 0,
                'AvgSpeed': float(section.find('vd:AvgSpd', ns).text) if section.find('vd:AvgSpd', ns) is not None and section.find('vd:AvgSpd', ns).text else 0.0,
                'AvgOccupancy': float(section.find('vd:AvgOcc', ns).text) if section.find('vd:AvgOcc', ns) is not None and section.find('vd:AvgOcc', ns).text else 0.0,
                'Svolume': 0,  # 小型車流量，XML中沒有此字段
                'Mvolume': 0,  # 中型車流量，XML中沒有此字段
                'Lvolume': 0,  # 大型車流量，XML中沒有此字段
            }
            sections.append(data)

        return sections

    except Exception as e:
        print(f"解析 XML 時發生錯誤: {e}")
        # 打印更多調試信息
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("開始運行腳本...")
    # 下載並保存數據
    xml_file = fetch_and_save_xml()
    print(f"下載結果: {xml_file}")

    if xml_file:
        print("載入映射文件...")
        # 載入設備映射
        device_mapping = load_device_mapping(os.path.join(script_dir, 'vd_sumo_mapping.csv'))
        print(f"載入映射完成，共 {len(device_mapping)} 個映射")

        print("解析XML數據...")
        # 解析數據
        traffic_data = parse_traffic_data(xml_file)
        print(f"解析完成，共 {len(traffic_data)} 條記錄")

        # 顯示前5個記錄的數據作為示例
        print("\n前5個記錄數據示例:")
        for i, data in enumerate(traffic_data[:5]):
            print(f"\n記錄 {i+1}:")
            for key, value in data.items():
                print(f"  {key}: {value}")

        print("處理數據並生成輸出...")
        # 合併數據並創建輸出
        if traffic_data:
            # 獲取當前時間戳
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 準備輸出數據
            output_data = []
            for data in traffic_data:
                device_id = data['DeviceID']
                sumo_edge_id = device_mapping.get(device_id, '')  # 如果找不到映射，留空

                output_record = {
                    'time': current_time,
                    'DeviceID': device_id,
                    'SumoEdgeID': sumo_edge_id,
                    'LaneNO': data['LaneNO'],
                    'Volume': data['Volume'],
                    'Svolume': data['Svolume'],
                    'Mvolume': data['Mvolume'],
                    'Lvolume': data['Lvolume']
                }
                output_data.append(output_record)

            print(f"準備輸出 {len(output_data)} 條記錄")

            # 保存為 vd_sumo_flow.csv
            csv_file = "vd_sumo_flow.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                if output_data:
                    writer = csv.DictWriter(f, fieldnames=output_data[0].keys())
                    writer.writeheader()
                    writer.writerows(output_data)

            print(f"\n數據已保存為 CSV 文件: {csv_file}")
            print(f"總共處理了 {len(output_data)} 條記錄")

            # 統計映射成功率
            mapped_count = sum(1 for record in output_data if record['SumoEdgeID'])
            print(f"設備映射成功率: {mapped_count}/{len(output_data)} ({mapped_count/len(output_data)*100:.1f}%)")
        else:
            print("\n沒有數據可保存")