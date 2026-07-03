"""
卡曼濾波車輛追蹤器（kalman_tracker.py）—— EKF + CTRV 運動模型
================================================================
回應「模擬中知道路線，現實怎麼辦」：現實中車輛只廣播 BSM/CAM
(位置/速度/航向，10Hz)。本模組只用這些「可觀測量」，以擴展卡曼濾波
(EKF) + 常數轉率與速度模型(CTRV)追蹤每台車的狀態：

    狀態 x = [px, py, v, θ, ω]      (ω = 轉彎率 rad/s，←關鍵：從航向歷史估出)
    量測 z = [px, py, v, θ]         (BSM/CAM 內容；不含 ω，ω 由濾波器推斷)

有了 ω 就能預判「對方會轉彎還是維持同向」：把兩車狀態依 CTRV 前推，
找出距離首次超過通訊半徑的時刻 = 預測連線壽命(Predicted Link Lifetime)。
文獻依據：Kalman 預測 VANET link residual lifetime(IEEE VTC 2016)、
EKF 位置預測(Wireless Pers. Commun. 2014)、PLL 指標(2026)。

純 numpy、不碰 SUMO；直接執行跑自我測試。
"""
import math
import numpy as np

# 量測雜訊(GNSS/BSM 等級)：位置 ~1m、速度 ~0.5m/s、航向 ~3度
R_MEAS = np.diag([1.0**2, 1.0**2, 0.5**2, math.radians(3.0)**2])
# 過程雜訊：允許加速/變向(否則濾波器跟不上機動)
Q_PROC = np.diag([0.3**2, 0.3**2, 0.8**2, math.radians(4.0)**2, 0.08**2])
# 前推預測時 ω 的衰減半衰期(秒)：轉彎是暫態機動，不會永遠轉圈
OMEGA_HALF_LIFE_S = 10.0


def _wrap(a):
    """把角度包回 [-π, π]。"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _ctrv_step(px, py, v, th, om, dt):
    """CTRV 前推一步(確定性)。"""
    if abs(om) > 1e-4:
        px += v / om * (math.sin(th + om * dt) - math.sin(th))
        py += v / om * (-math.cos(th + om * dt) + math.cos(th))
    else:
        px += v * math.cos(th) * dt
        py += v * math.sin(th) * dt
    return px, py, v, _wrap(th + om * dt), om


class EKFCTRV:
    """單車 EKF-CTRV 追蹤器：餵入逐秒 BSM 量測，維護含轉彎率的狀態估計。"""

    def __init__(self):
        self.x = None            # [px, py, v, θ, ω]
        self.P = None
        self.n_updates = 0

    # ---- 對外屬性 ----
    @property
    def ready(self):
        """至少 3 筆量測後 ω 估計才有意義。"""
        return self.n_updates >= 3

    @property
    def omega(self):
        return float(self.x[4]) if self.x is not None else 0.0

    # ---- 濾波 ----
    def step(self, px, py, speed, theta, dt):
        """一個 tick：predict(dt) + update(量測)。theta 為數學慣例(x軸起、逆時針、弧度)。"""
        z = np.array([px, py, speed, theta], dtype=float)
        if self.x is None:       # 首筆量測：直接初始化，ω 未知給大變異
            self.x = np.array([px, py, speed, theta, 0.0])
            self.P = np.diag([2.0, 2.0, 1.0, 0.2, 0.5])
            self.n_updates = 1
            return
        self._predict(dt)
        self._update(z)
        self.n_updates += 1

    def _predict(self, dt):
        px, py, v, th, om = self.x
        self.x = np.array(_ctrv_step(px, py, v, th, om, dt))
        # Jacobian F
        F = np.eye(5)
        if abs(om) > 1e-4:
            s0, s1 = math.sin(th), math.sin(th + om * dt)
            c0, c1 = math.cos(th), math.cos(th + om * dt)
            F[0, 2] = (s1 - s0) / om
            F[0, 3] = v / om * (c1 - c0)
            F[0, 4] = v * (dt * c1 * om - (s1 - s0)) / om**2
            F[1, 2] = (c0 - c1) / om
            F[1, 3] = v / om * (s1 - s0)
            F[1, 4] = v * (dt * s1 * om - (c0 - c1)) / om**2
        else:
            F[0, 2] = math.cos(th) * dt
            F[0, 3] = -v * math.sin(th) * dt
            F[1, 2] = math.sin(th) * dt
            F[1, 3] = v * math.cos(th) * dt
        F[3, 4] = dt
        self.P = F @ self.P @ F.T + Q_PROC * dt

    def _update(self, z):
        H = np.zeros((4, 5))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        y = z - H @ self.x
        y[3] = _wrap(y[3])                       # 航向殘差要包角(反彈/迴轉時 ±π)
        S = H @ self.P @ H.T + R_MEAS
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[3] = _wrap(self.x[3])
        self.P = (np.eye(5) - K @ H) @ self.P

    # ---- 預測 ----
    def forward_position(self, t, dt=0.5):
        """依目前估計(含 ω 衰減)前推 t 秒後的位置。"""
        px, py, v, th, om = self.x
        decay = 0.5 ** (dt / OMEGA_HALF_LIFE_S)
        steps = max(1, int(round(t / dt)))
        for _ in range(steps):
            px, py, v, th, om = _ctrv_step(px, py, v, th, om, dt)
            om *= decay
        return px, py


def predict_contact(tr_a, tr_b, comm_range, horizon=60.0, dt=0.5):
    """
    兩台已追蹤車輛的「預測連線壽命」(秒)：把兩車 CTRV 前推，
    回傳距離首次超過 comm_range 的時刻；期間都在範圍內 → horizon。
    ω 以半衰期衰減(轉彎為暫態，避免預測出永久繞圈)。
    """
    a = list(tr_a.x)
    b = list(tr_b.x)
    decay = 0.5 ** (dt / OMEGA_HALF_LIFE_S)
    if math.dist(a[:2], b[:2]) > comm_range:
        return 0.0
    t = 0.0
    while t < horizon:
        a = list(_ctrv_step(*a, dt)); a[4] *= decay
        b = list(_ctrv_step(*b, dt)); b[4] *= decay
        t += dt
        if math.dist(a[:2], b[:2]) > comm_range:
            return t
    return horizon


# ==================================================================
# 自我測試：python kalman_tracker.py
# ==================================================================
if __name__ == "__main__":
    print("=== EKF-CTRV 自我測試 ===\n")

    # [1] 直行車：5 秒後位置預測應接近真值
    tr = EKFCTRV()
    for k in range(10):
        tr.step(10.0 * k, 0.0, 10.0, 0.0, 1.0)
    px, py = tr.forward_position(5.0)
    print(f"[1] 直行車(10m/s) 5s 後預測 ({px:.1f},{py:.1f})，真值 (140,0)"
          f" → 誤差 {math.dist((px,py),(140,0)):.1f} m")

    # [2] 定轉率車：ω 估計應收斂到真值 0.2 rad/s
    om_true, v, th = 0.2, 10.0, 0.0
    x = y = 0.0
    tr2 = EKFCTRV()
    for k in range(15):
        tr2.step(x, y, v, th, 1.0)
        x, y, v, th, _ = _ctrv_step(x, y, v, th, om_true, 1.0)
    print(f"[2] 轉彎車 ω 真值 0.200 → 估計 {tr2.omega:.3f}"
          f"（|誤差| {abs(tr2.omega-om_true):.3f}，應 <0.05）")

    # [3] 轉彎預判：B 開始右轉 → kalman contact 應短於「B 直行」情境
    def feed(turn_last_k):
        trA, trB = EKFCTRV(), EKFCTRV()
        xB, yB, thB = 0.0, 30.0, 0.0
        for k in range(12):
            trA.step(10.0 * k, 0.0, 10.0, 0.0, 1.0)
            trB.step(xB, yB, 10.0, thB, 1.0)
            om = 0.25 if k >= 12 - turn_last_k else 0.0
            xB, yB, _, thB, _ = _ctrv_step(xB, yB, 10.0, thB, om, 1.0)
        return predict_contact(trA, trB, 150.0)
    c_straight = feed(0)
    c_turning = feed(4)
    print(f"[3] 同向同速 contact={c_straight:.1f}s；對方開始轉彎 → {c_turning:.1f}s"
          f"（轉彎應明顯較短）")

    ok = (math.dist((px, py), (140, 0)) < 5 and abs(tr2.omega - om_true) < 0.05
          and c_turning < c_straight * 0.7)
    print("\n=== " + ("全部通過 ✓" if ok else "有項目未達標 ✗") + " ===")
    raise SystemExit(0 if ok else 1)
