@echo off
REM ============================================================
REM build_scenario.bat —— 把「純路網」補成可跑的模擬場景
REM   產生：合成車流(randomTrips) + RSU(setup_rsu) + osm.sumocfg
REM 用法：  build_scenario.bat new.net.xml
REM         build_scenario.bat new.net.xml 2.0     (第2參數=車輛發車間隔秒，越小車越多)
REM 產出：  scenario.rou.xml、rsu.add.xml、rsu_positions.json、osm.sumocfg
REM 之後：  python train_mappo.py --sumo   (會自動讀 osm.sumocfg)
REM ============================================================
if "%~1"=="" ( echo 用法: build_scenario.bat ^<net.xml^> [發車間隔秒] & exit /b 1 )
if not defined SUMO_HOME ( echo [錯誤] 未設定 SUMO_HOME & exit /b 1 )
set NET=%~1
set PERIOD=%~2
if "%PERIOD%"=="" set PERIOD=2.0

echo === 1/3 產生合成車流(randomTrips，發車間隔 %PERIOD%s、模擬 3600s) ===
python "%SUMO_HOME%\tools\randomTrips.py" -n "%NET%" -r scenario.rou.xml ^
  -e 3600 -p %PERIOD% --fringe-factor 5 --validate --seed 42
if errorlevel 1 ( echo [失敗] randomTrips 出錯 & exit /b 1 )

echo === 2/3 佈署邊緣基站 RSU(路口四角+路肩補洞) ===
python setup_rsu.py --net "%NET%" --mode junction --corners 2 --plot
if errorlevel 1 ( echo [失敗] setup_rsu 出錯 & exit /b 1 )

echo === 3/3 產生 osm.sumocfg(綁定 網路+車流+RSU) ===
> osm.sumocfg echo ^<configuration^>
>> osm.sumocfg echo   ^<input^>
>> osm.sumocfg echo     ^<net-file value="%NET%"/^>
>> osm.sumocfg echo     ^<route-files value="scenario.rou.xml"/^>
>> osm.sumocfg echo     ^<additional-files value="rsu.add.xml"/^>
>> osm.sumocfg echo   ^</input^>
>> osm.sumocfg echo   ^<time^>^<begin value="0"/^>^<end value="3600"/^>^</time^>
>> osm.sumocfg echo ^</configuration^>

echo.
echo === 完成 ===
echo   場景檔: osm.sumocfg
echo   預覽:   sumo-gui -c osm.sumocfg   (應看到車在跑 + 藍點 RSU)
echo   訓練:   python train_mappo.py --sumo
