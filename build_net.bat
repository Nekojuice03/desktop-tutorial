@echo off
REM ============================================================
REM build_net.bat —— 不靠 osmWebWizard，把 OSM 檔轉成 SUMO 路網
REM 用法：  build_net.bat my_area.osm     (支援 .osm / .osm.xml / .osm.gz)
REM 產出：  new.net.xml(路網)、new.poly.xml(建物，僅視覺)
REM 參數與本專案 osm.netccfg(wizard 產生)一致 → 與 wizard 轉檔等效。
REM 轉完接著跑：python setup_rsu.py  (自適應重佈 RSU)
REM ============================================================
if "%~1"=="" (
  echo 用法: build_net.bat ^<osm檔^>   例如: build_net.bat daan.osm
  exit /b 1
)
if not defined SUMO_HOME (
  echo [錯誤] 未設定 SUMO_HOME，請先: set SUMO_HOME=C:\Program Files ^(x86^)\Eclipse\Sumo
  exit /b 1
)

echo === netconvert：OSM -> SUMO 路網 ===
netconvert --osm-files "%~1" ^
  -t "%SUMO_HOME%\data\typemap\osmNetconvert.typ.xml" ^
  --output-file new.net.xml ^
  --output.street-names true --output.original-names true ^
  --geometry.remove --ramps.guess --junctions.join ^
  --tls.discard-simple --tls.join --tls.guess-signals ^
  --tls.default-type actuated ^
  --keep-edges.by-vclass passenger
if errorlevel 1 ( echo [失敗] netconvert 出錯 & exit /b 1 )

echo === polyconvert：建物多邊形(視覺用，可略) ===
polyconvert --net-file new.net.xml --osm-files "%~1" ^
  -t "%SUMO_HOME%\data\typemap\osmPolyconvert.typ.xml" ^
  --output-file new.poly.xml
if errorlevel 1 echo [提醒] polyconvert 失敗不影響路網，可忽略

echo.
echo === 完成 ===
echo   路網: new.net.xml   建物: new.poly.xml
echo   檢視: sumo-gui -n new.net.xml
echo   下一步: 把 new.net.xml 換名/放進專案，執行 python setup_rsu.py 重佈 RSU
