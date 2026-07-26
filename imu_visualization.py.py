from __future__ import annotations

import argparse
import csv
import math
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import serial
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *



G = np.array([0.0, 0.0, -9.80665], dtype=float)

ACC_SCALE = 9.80665 / 16384.0
GYRO_SCALE = math.pi / 180.0 / 131.0

CALIBRATION_SAMPLES = 250
MIN_DT = 1e-4
MAX_DT = 0.02
GRAVITY_CORRECTION_GAIN = 2.0
VELOCITY_DAMPING = 0.980
ACC_DEADBAND = 0.22
STILL_WINDOW_SEC = 0.35
RESET_COOLDOWN_SEC = 0.40
POSITION_DISPLAY_GAIN = 4.0


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array(
        [[0.0, -z, y],
         [z, 0.0, -x],
         [-y, x, 0.0]],
        dtype=float
    )


def exp_so3(phi: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(phi))
    K = skew(phi)

    if theta < 1e-8:
        return np.eye(3) + K

    return (
        np.eye(3)
        + math.sin(theta) / theta * K
        + (1.0 - math.cos(theta)) / (theta * theta) * (K @ K)
    )


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize(a)
    b = normalize(b)

    c = float(np.clip(np.dot(a, b), -1.0, 1.0))

    if c > 1.0 - 1e-8:
        return np.eye(3)

    if c < -1.0 + 1e-8:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        return exp_so3(normalize(axis) * math.pi)

    axis = normalize(np.cross(a, b))
    angle = math.acos(c)
    return exp_so3(axis * angle)


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array(
        [[1, 0, 0],
         [0, cr, -sr],
         [0, sr, cr]],
        dtype=float
    )

    Ry = np.array(
        [[cp, 0, sp],
         [0, 1, 0],
         [-sp, 0, cp]],
        dtype=float
    )

    Rz = np.array(
        [[cy, -sy, 0],
         [sy, cy, 0],
         [0, 0, 1]],
        dtype=float
    )

    return Rz @ Ry @ Rx


def rot_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


@dataclass
class ImuSample:
    ax: int
    ay: int
    az: int
    gx: int
    gy: int
    gz: int
    t: float


class SerialWorker(threading.Thread):
    def __init__(self, port: str, baud: int, out_q: queue.Queue[ImuSample]):
        super().__init__(daemon=True)

        if serial is None:
            raise RuntimeError("pyserial is not installed. Run: pip install pyserial")

        self.port = port
        self.baud = baud
        self.out_q = out_q
        self.running = True
        self.ser: Optional[serial.Serial] = None

    def run(self) -> None:
        self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
        time.sleep(2.0)
        self.ser.reset_input_buffer()

        while self.running:
            try:
                line = self.ser.readline().decode("ascii", errors="ignore").strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")

                if len(parts) != 6:
                    continue

                values = [int(p.strip()) for p in parts]
                sample = ImuSample(*values, time.time())

                if self.out_q.full():
                    try:
                        self.out_q.get_nowait()
                    except queue.Empty:
                        pass

                self.out_q.put_nowait(sample)

            except Exception:
                continue

    def stop(self) -> None:
        self.running = False

        if self.ser is not None:
            self.ser.close()


class DemoSource(QtCore.QObject):
    new_sample = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.t0 = time.time()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.emit_sample)
        self.timer.start(2)

    def emit_sample(self) -> None:
        t = time.time() - self.t0

        roll = 0.55 * math.sin(2.0 * t)
        pitch = 0.40 * math.sin(2.7 * t + 0.3)
        yaw = 0.75 * math.sin(1.5 * t)

        R = rpy_to_rot(roll, pitch, yaw)

        a_world = np.array(
            [
                0.45 * math.sin(2.1 * t),
                0.30 * math.sin(3.0 * t),
                0.20 * math.sin(1.7 * t),
            ],
            dtype=float
        )

        f_body = R.T @ (a_world - G)

        gyro_body = np.array(
            [
                0.55 * 2.0 * math.cos(2.0 * t),
                0.40 * 2.7 * math.cos(2.7 * t + 0.3),
                0.75 * 1.5 * math.cos(1.5 * t),
            ],
            dtype=float
        )

        acc_raw = np.round(f_body / ACC_SCALE).astype(int)
        gyro_raw = np.round(gyro_body / GYRO_SCALE).astype(int)

        self.new_sample.emit(
            ImuSample(
                acc_raw[0],
                acc_raw[1],
                acc_raw[2],
                gyro_raw[0],
                gyro_raw[1],
                gyro_raw[2],
                time.time(),
            )
        )


class CsvLogger:
    def __init__(self, path: Optional[str]):
        self.file = None
        self.writer = None

        if path:
            self.file = open(path, "w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.file)
            self.writer.writerow(
                [
                    "time",
                    "ax",
                    "ay",
                    "az",
                    "gx",
                    "gy",
                    "gz",
                    "px",
                    "py",
                    "pz",
                    "roll",
                    "pitch",
                    "yaw",
                    "calibrated",
                ]
            )

    def write(self, s: ImuSample, p: np.ndarray, R: np.ndarray, calibrated: bool) -> None:
        if self.writer is None:
            return

        roll, pitch, yaw = rot_to_rpy(R)

        self.writer.writerow(
            [
                s.t,
                s.ax,
                s.ay,
                s.az,
                s.gx,
                s.gy,
                s.gz,
                p[0],
                p[1],
                p[2],
                roll,
                pitch,
                yaw,
                int(calibrated),
            ]
        )

    def close(self) -> None:
        if self.file:
            self.file.close()


class ImuEstimator:
    def __init__(self):
        self.R = np.eye(3)
        self.v = np.zeros(3)
        self.p = np.zeros(3)

        self.acc_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

        self.calibrated = False

        self.acc_calib = []
        self.gyro_calib = []

        self.last_t: Optional[float] = None

        self.still_buffer = deque()
        self.last_reset_t = 0.0

    def raw_to_units(self, s: ImuSample) -> Tuple[np.ndarray, np.ndarray]:
        acc = np.array([s.ax, s.ay, s.az], dtype=float) * ACC_SCALE
        gyro = np.array([s.gx, s.gy, s.gz], dtype=float) * GYRO_SCALE
        return acc, gyro

    def calibrate(self, acc: np.ndarray, gyro: np.ndarray) -> bool:
        if self.calibrated:
            return True

        self.acc_calib.append(acc)
        self.gyro_calib.append(gyro)

        if len(self.acc_calib) < CALIBRATION_SAMPLES:
            return False

        acc_mean = np.mean(np.asarray(self.acc_calib), axis=0)
        gyro_mean = np.mean(np.asarray(self.gyro_calib), axis=0)

        self.R = rot_between(acc_mean, -G)
        self.gyro_bias = gyro_mean
        self.acc_bias = acc_mean - self.R.T @ (-G)

        self.v[:] = 0.0
        self.p[:] = 0.0

        self.calibrated = True

        return True

    def update_still_buffer(self, t: float, acc: np.ndarray, gyro: np.ndarray) -> None:
        self.still_buffer.append((t, acc.copy(), gyro.copy()))

        while self.still_buffer and t - self.still_buffer[0][0] > STILL_WINDOW_SEC:
            self.still_buffer.popleft()

    def auto_reset_if_still(self, t: float) -> None:
        if len(self.still_buffer) < 25:
            return

        if self.still_buffer[-1][0] - self.still_buffer[0][0] < 0.8 * STILL_WINDOW_SEC:
            return

        accs = np.asarray([x[1] for x in self.still_buffer])
        gyros = np.asarray([x[2] for x in self.still_buffer])

        gyro_level = float(np.max(np.linalg.norm(gyros - self.gyro_bias, axis=1)))
        acc_range = float(np.max(np.max(accs, axis=0) - np.min(accs, axis=0)))

        if (
            gyro_level < math.radians(6.0)
            and acc_range < 0.45
            and t - self.last_reset_t > RESET_COOLDOWN_SEC
        ):
            acc_mean = np.mean(accs, axis=0)
            gyro_mean = np.mean(gyros, axis=0)

            self.R = rot_between(acc_mean, -G)
            self.gyro_bias = gyro_mean
            self.acc_bias = acc_mean - self.R.T @ (-G)

            self.v[:] = 0.0
            self.p[:] = 0.0

            self.last_reset_t = t

    def step(self, sample: ImuSample) -> Tuple[np.ndarray, np.ndarray, bool]:
        acc_m, gyro_m = self.raw_to_units(sample)

        self.update_still_buffer(sample.t, acc_m, gyro_m)

        if not self.calibrate(acc_m, gyro_m):
            return self.p, self.R, False

        if self.last_t is None:
            self.last_t = sample.t
            return self.p, self.R, True

        dt = min(max(sample.t - self.last_t, MIN_DT), MAX_DT)
        self.last_t = sample.t

        acc = acc_m - self.acc_bias
        gyro = gyro_m - self.gyro_bias

        self.R = self.R @ exp_so3(gyro * dt)

        if abs(np.linalg.norm(acc) - 9.80665) < 0.9 and np.linalg.norm(gyro) < math.radians(30.0):
            measured_up = normalize(self.R @ normalize(acc))
            true_up = normalize(-G)
            error_axis = np.cross(measured_up, true_up)
            correction = min(GRAVITY_CORRECTION_GAIN * dt, 0.06) * error_axis
            self.R = exp_so3(correction) @ self.R

        a_world = self.R @ acc + G

        if np.linalg.norm(a_world) < ACC_DEADBAND and np.linalg.norm(gyro) < math.radians(5.0):
            a_world[:] = 0.0
            self.v[:] = 0.0
            self.p[:] = 0.0
        elif np.linalg.norm(a_world) < ACC_DEADBAND:
            a_world[:] = 0.0
        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = VELOCITY_DAMPING * (self.v + a_world * dt)

        self.auto_reset_if_still(sample.t)

        return self.p, self.R, True


class Viewer(QOpenGLWidget):
    def __init__(self):
        super().__init__()

        self.setMinimumSize(1000, 700)

        self.p = np.zeros(3)
        self.R = np.eye(3)
        self.calibrated = False

        self.path = deque(maxlen=600)

    def set_state(self, p: np.ndarray, R: np.ndarray, calibrated: bool) -> None:
        self.p = p * POSITION_DISPLAY_GAIN
        self.R = R.copy()
        self.calibrated = calibrated

        if calibrated:
            self.path.append(self.p.copy())

        self.update()

    def initializeGL(self) -> None:
        glClearColor(0.08, 0.09, 0.11, 1.0)
        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, max(1, h))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / max(1, h), 0.01, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        gluLookAt(3.2, -5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        self.draw_grid()
        self.draw_axes()
        self.draw_path()
        self.draw_platform()

    def draw_grid(self) -> None:
        glColor3f(0.35, 0.35, 0.35)

        glBegin(GL_LINES)

        for i in range(-10, 11):
            x = i * 0.2

            glVertex3f(x, -2.0, 0.0)
            glVertex3f(x, 2.0, 0.0)

            glVertex3f(-2.0, x, 0.0)
            glVertex3f(2.0, x, 0.0)

        glEnd()

    def draw_axes(self) -> None:
        glLineWidth(2.5)

        glBegin(GL_LINES)

        glColor3f(1.0, 0.2, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(0.8, 0, 0)

        glColor3f(0.2, 1.0, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0.8, 0)

        glColor3f(0.2, 0.4, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 0.8)

        glEnd()

        glLineWidth(1.0)

    def draw_path(self) -> None:
        if len(self.path) < 2:
            return

        glColor3f(1.0, 0.8, 0.1)

        glBegin(GL_LINE_STRIP)

        for p in self.path:
            glVertex3f(float(p[0]), float(p[1]), float(p[2]))

        glEnd()

    def draw_box(self, lx: float, ly: float, lz: float) -> None:
        x = lx / 2.0
        y = ly / 2.0
        z = lz / 2.0

        glBegin(GL_QUADS)

        glVertex3f(-x, -y, z)
        glVertex3f(x, -y, z)
        glVertex3f(x, y, z)
        glVertex3f(-x, y, z)

        glVertex3f(-x, -y, -z)
        glVertex3f(-x, y, -z)
        glVertex3f(x, y, -z)
        glVertex3f(x, -y, -z)

        glVertex3f(-x, y, -z)
        glVertex3f(-x, y, z)
        glVertex3f(x, y, z)
        glVertex3f(x, y, -z)

        glVertex3f(-x, -y, -z)
        glVertex3f(x, -y, -z)
        glVertex3f(x, -y, z)
        glVertex3f(-x, -y, z)

        glVertex3f(x, -y, -z)
        glVertex3f(x, y, -z)
        glVertex3f(x, y, z)
        glVertex3f(x, -y, z)

        glVertex3f(-x, -y, -z)
        glVertex3f(-x, -y, z)
        glVertex3f(-x, y, z)
        glVertex3f(-x, y, -z)

        glEnd()

    def draw_rotor(self, x: float, y: float) -> None:
        glPushMatrix()
        glTranslatef(x, y, 0.02)

        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0.0, 0.0, 0.0)

        for k in range(33):
            a = 2.0 * math.pi * k / 32.0
            glVertex3f(0.12 * math.cos(a), 0.12 * math.sin(a), 0.0)

        glEnd()

        glColor3f(0.12, 0.12, 0.12)

        glBegin(GL_LINES)

        glVertex3f(-0.18, 0.0, 0.01)
        glVertex3f(0.18, 0.0, 0.01)

        glVertex3f(0.0, -0.18, 0.01)
        glVertex3f(0.0, 0.18, 0.01)

        glEnd()

        glPopMatrix()

    def draw_platform(self) -> None:
        glPushMatrix()

        glTranslatef(float(self.p[0]), float(self.p[1]), float(self.p[2]))

        roll, pitch, yaw = rot_to_rpy(self.R)
        display_R = rpy_to_rot(-roll, -pitch, yaw)

        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = display_R

        glMultMatrixf(M.T)

        glColor3f(0.18, 0.68, 0.75)
        self.draw_box(0.42, 0.24, 0.12)

        glLineWidth(5.0)
        glColor3f(0.78, 0.78, 0.78)

        glBegin(GL_LINES)

        glVertex3f(-0.55, 0.0, 0.0)
        glVertex3f(0.55, 0.0, 0.0)

        glVertex3f(0.0, -0.55, 0.0)
        glVertex3f(0.0, 0.55, 0.0)

        glEnd()

        glLineWidth(1.0)

        glColor3f(0.96, 0.56, 0.20)
        self.draw_rotor(0.55, 0.0)

        glColor3f(0.44, 0.76, 0.28)
        self.draw_rotor(-0.55, 0.0)

        glColor3f(0.70, 0.35, 0.85)
        self.draw_rotor(0.0, 0.55)

        glColor3f(0.95, 0.80, 0.25)
        self.draw_rotor(0.0, -0.55)

        glColor3f(1.0, 1.0, 1.0)

        glBegin(GL_TRIANGLES)

        glVertex3f(0.32, 0.0, 0.09)
        glVertex3f(0.12, 0.08, 0.09)
        glVertex3f(0.12, -0.08, 0.09)

        glEnd()

        glPopMatrix()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()

        self.setWindowTitle("Real-Time IMU Motion Viewer")

        self.viewer = Viewer()
        self.setCentralWidget(self.viewer)

        self.estimator = ImuEstimator()

        self.samples: queue.Queue[ImuSample] = queue.Queue(maxsize=3000)

        self.reader: Optional[SerialWorker] = None
        self.demo: Optional[DemoSource] = None

        self.logger = CsvLogger(args.log)

        if args.demo:
            self.demo = DemoSource()
            self.demo.new_sample.connect(self.handle_sample)

        else:
            if not args.port:
                raise ValueError("Please provide --port, for example --port COM3")

            self.reader = SerialWorker(args.port, args.baud, self.samples)
            self.reader.start()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.consume_samples)
        self.timer.start(16)

        self.statusBar().showMessage("Keep the IMU still for calibration...")

    def consume_samples(self) -> None:
        for _ in range(250):
            try:
                sample = self.samples.get_nowait()
            except queue.Empty:
                break

            self.handle_sample(sample)

    def handle_sample(self, sample: ImuSample) -> None:
        p, R, calibrated = self.estimator.step(sample)

        self.viewer.set_state(p, R, calibrated)
        self.logger.write(sample, p, R, calibrated)

        if calibrated:
            roll, pitch, yaw = rot_to_rpy(R)

            self.statusBar().showMessage(
                f"calibrated | p=[{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}] m | "
                f"rpy=[{math.degrees(roll):+.1f}, {math.degrees(pitch):+.1f}, {math.degrees(yaw):+.1f}] deg"
            )

        else:
            n = len(self.estimator.acc_calib)
            self.statusBar().showMessage(f"calibrating {n}/{CALIBRATION_SAMPLES}: keep the MPU6050 still")

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key.Key_S:
            name = time.strftime("imu_snapshot_%Y%m%d_%H%M%S.png")
            self.viewer.grabFramebuffer().save(name)
            self.statusBar().showMessage(f"saved {name}")

        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self.reader is not None:
            self.reader.stop()

        self.logger.close()

        super().closeEvent(event)


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time IMU motion visualization")

    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--baud", type=int, default=500000)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--log", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app = QtWidgets.QApplication(sys.argv)

    win = MainWindow(args)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()