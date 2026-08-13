import os
import time
import numpy as np
import cv2
import onnxruntime as ort
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QTextEdit, QGroupBox, QProgressBar, QMessageBox,
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ==================== 常量配置 ====================
ONNX_PATH = "./checkpoints/mflnet_20260810_170418/best.onnx"
INPUT_SIZE = 1024            # 推理输入尺寸
RESULT_DIR = "./results/ui_output"  # 批量分割默认保存目录

# 颜色映射（BGR格式，与OpenCV一致）
COLOR_MAP = {
    0: [128, 0, 0],         # 背景
    1: [0, 0, 255],         # 建筑
    2: [0, 255, 0],         # 道路
    3: [255, 0, 0],         # 水体
    4: [0, 255, 255],       # 裸地
    5: [255, 0, 255],       # 森林
    6: [255, 255, 0]        # 农业土地
}

# 类别名称映射
NAME_MAP = {
    0: "背景",
    1: "建筑",
    2: "道路",
    3: "水体",
    4: "裸地",
    5: "森林",
    6: "农业土地"
}

# ImageNet归一化参数（与训练一致）
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')


# ==================== 后端：模型推理与结果处理 ====================
class SegmentationBackend:
    """语义分割后端：负责ONNX模型加载、推理预测、颜色映射与结果保存"""

    def __init__(self, onnx_path=ONNX_PATH):
        self.session = None
        self.input_name = None
        self.load_model(onnx_path)

    def load_model(self, onnx_path):
        """加载ONNX推理会话，检测并优先使用GPU"""
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX模型文件不存在: {onnx_path}")
        # 检查可用Provider，优先CUDA
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            try:
                self.session = ort.InferenceSession(
                    onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
            except Exception:
                self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        else:
            self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        # 记录实际使用的推理设备
        self.device = 'GPU' if 'CUDAExecutionProvider' in self.session.get_providers() else 'CPU'

    def preprocess(self, image_rgb):
        """图像预处理：Resize + 归一化 + 转NCHW float32"""
        resized = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        img = resized.astype(np.float32) / 255.0
        img = (img - NORM_MEAN) / NORM_STD
        img = np.transpose(img, (2, 0, 1))       # HWC -> CHW
        img = np.expand_dims(img, axis=0)         # CHW -> NCHW
        return np.ascontiguousarray(img)

    def inference(self, input_tensor):
        """ONNX前向推理，返回logits输出"""
        output = self.session.run(None, {self.input_name: input_tensor})
        return output[0]

    def postprocess(self, output, orig_size):
        """后处理：argmax得到类别，最近邻还原到原始尺寸"""
        pred = np.argmax(output[0], axis=0).astype(np.uint8)  # (H, W)
        pred = cv2.resize(pred, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
        return pred

    def apply_colormap(self, pred_mask):
        """将单通道预测掩码映射为RGB彩色图（COLOR_MAP为BGR，此处转为RGB用于显示）"""
        h, w = pred_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, bgr in COLOR_MAP.items():
            colored[pred_mask == cls_id] = bgr[::-1]  # BGR -> RGB
        return colored

    def compute_stats(self, pred_mask):
        """统计各类别像素数与面积占比"""
        total = pred_mask.size
        stats = {}
        for cls_id in sorted(COLOR_MAP.keys()):
            count = int(np.sum(pred_mask == cls_id))
            stats[cls_id] = {'count': count, 'ratio': count / total * 100}
        return stats

    def predict(self, image_path):
        """完整预测流程：加载图片 -> 预处理 -> 推理 -> 后处理，返回结果字典"""
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"无法读取图片: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        # 预处理 + 推理（计时）
        input_tensor = self.preprocess(image_rgb)
        t0 = time.perf_counter()
        output = self.inference(input_tensor)
        infer_time = (time.perf_counter() - t0) * 1000  # 毫秒

        # 后处理
        pred_mask = self.postprocess(output, (orig_h, orig_w))
        colored_mask = self.apply_colormap(pred_mask)
        stats = self.compute_stats(pred_mask)

        return {
            'image_rgb': image_rgb,
            'pred_mask': pred_mask,
            'colored_mask': colored_mask,
            'stats': stats,
            'infer_time': infer_time,
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
        }

    @staticmethod
    def save_mask(pred_mask, save_path):
        """保存单通道掩码"""
        cv2.imwrite(save_path, pred_mask)

    @staticmethod
    def save_colored(colored_mask, save_path):
        """保存彩色分割结果（RGB -> BGR for cv2）"""
        cv2.imwrite(save_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))


# ==================== 推理工作线程（避免阻塞UI） ====================
class PredictWorker(QThread):
    """后台推理线程，逐张处理图片并通过信号回传结果"""
    result_ready = pyqtSignal(dict)       # 单张结果
    progress = pyqtSignal(int, int)       # (当前, 总数)
    finished_all = pyqtSignal()           # 全部完成
    error = pyqtSignal(str)               # 错误信息

    def __init__(self, backend, image_paths):
        super().__init__()
        self.backend = backend
        self.image_paths = image_paths

    def run(self):
        total = len(self.image_paths)
        for i, path in enumerate(self.image_paths):
            try:
                result = self.backend.predict(path)
                self.result_ready.emit(result)
            except Exception as e:
                self.error.emit(f"{os.path.basename(path)}: {e}")
            self.progress.emit(i + 1, total)
        self.finished_all.emit()


# ==================== 前端：主窗口 ====================
class MainWindow(QMainWindow):
    """主窗口：左侧按钮 | 中间双图+信息 | 右侧图例"""

    def __init__(self):
        super().__init__()
        self.backend = None
        self.worker = None
        self.current_image_path = None
        self.current_colored_mask = None    # 缓存彩色结果，用于resize重绘
        self.folder_image_paths = []
        self.batch_results = []
        self.save_dir = None                # 结果保存路径

        self.init_ui()
        self.init_backend()
        self.apply_style()

    # ---- 初始化 ----
    def init_backend(self):
        """初始化后端，加载ONNX模型并显示推理设备"""
        try:
            self.backend = SegmentationBackend(ONNX_PATH)
            device = getattr(self.backend, 'device', 'CPU')
            self.lbl_status.setText(f"模型已加载 | 推理设备: {device}")
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", str(e))

    def init_ui(self):
        self.setWindowTitle("遥感图像语义分割系统")
        self.setMinimumSize(1200, 750)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        main_layout.addLayout(self._build_button_panel(), 0)
        main_layout.addLayout(self._build_display_area(), 1)
        main_layout.addWidget(self._build_legend(), 0)

    def _build_button_panel(self):
        """构建左侧按钮面板"""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        title = QLabel("功能面板")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)

        self.btn_open_image = QPushButton("打开图片")
        self.btn_open_folder = QPushButton("打开文件夹")
        self.btn_segment = QPushButton("图片分割")
        self.btn_batch = QPushButton("批量分割")
        self.btn_save_dir = QPushButton("选择保存路径")

        for btn in (self.btn_open_image, self.btn_open_folder, self.btn_segment,
                     self.btn_batch, self.btn_save_dir):
            btn.setMinimumHeight(40)
            btn.setCursor(Qt.PointingHandCursor)
            layout.addWidget(btn)

        # 保存路径显示标签
        self.lbl_save_path = QLabel("保存路径: 未选择")
        self.lbl_save_path.setWordWrap(True)
        self.lbl_save_path.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 2px;")
        layout.addWidget(self.lbl_save_path)

        # 状态标签
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #7f8c8d; font-size: 12px; padding: 4px;")
        layout.addWidget(self.lbl_status)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 文件计数标签
        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet("color: #95a5a6; font-size: 11px;")
        layout.addWidget(self.lbl_count)

        layout.addStretch()

        self.btn_open_image.clicked.connect(self.open_image)
        self.btn_open_folder.clicked.connect(self.open_folder)
        self.btn_segment.clicked.connect(self.segment_image)
        self.btn_batch.clicked.connect(self.batch_segment)
        self.btn_save_dir.clicked.connect(self.select_save_dir)
        return layout

    def _build_display_area(self):
        """构建中间显示区：上方双图 + 下方信息"""
        layout = QVBoxLayout()
        layout.setSpacing(6)

        # 上方：原图 | 分割结果
        img_layout = QHBoxLayout()
        img_layout.setSpacing(6)

        self.lbl_original = QLabel("未加载图片")
        self.lbl_result = QLabel("未加载图片")
        for lbl in (self.lbl_original, self.lbl_result):
            lbl.setMinimumSize(450, 350)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "background-color: #d5d8dc; color: #7f8c8d; "
                "font-size: 14px; border: 1px solid #bdc3c7; border-radius: 4px;"
            )
        img_layout.addWidget(self.lbl_original, 1)
        img_layout.addWidget(self.lbl_result, 1)
        layout.addLayout(img_layout, 1)

        # 下方：分割信息
        info_group = QGroupBox("分割结果信息")
        info_layout = QVBoxLayout()
        self.txt_info = QTextEdit()
        self.txt_info.setReadOnly(True)
        self.txt_info.setMaximumHeight(160)
        self.txt_info.setStyleSheet("font-size: 13px; background-color: #ffffff;")
        info_layout.addWidget(self.txt_info)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        return layout

    def _build_legend(self):
        """构建右侧颜色图例"""
        group = QGroupBox("类别图例")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        for cls_id in sorted(COLOR_MAP.keys()):
            row = QHBoxLayout()
            bgr = COLOR_MAP[cls_id]
            r, g, b = bgr[2], bgr[1], bgr[0]  # BGR -> RGB
            swatch = QLabel()
            swatch.setFixedSize(24, 24)
            swatch.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #2c3e50; border-radius: 3px;"
            )
            row.addWidget(swatch)
            name = QLabel(NAME_MAP[cls_id])
            name.setStyleSheet("font-size: 13px;")
            row.addWidget(name)
            row.addStretch()
            layout.addLayout(row)

        layout.addStretch()
        group.setLayout(layout)
        group.setFixedWidth(160)
        return group

    def apply_style(self):
        """应用全局QSS样式"""
        self.setStyleSheet("""
            QMainWindow { background-color: #ecf0f1; }
            QPushButton {
                background-color: #3498db; color: white; border: none;
                border-radius: 5px; font-size: 14px; padding: 8px 16px;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:pressed { background-color: #21618c; }
            QPushButton:disabled { background-color: #bdc3c7; }
            QPushButton#btnSegment { background-color: #27ae60; }
            QPushButton#btnSegment:hover { background-color: #229954; }
            QPushButton#btnBatch { background-color: #e67e22; }
            QPushButton#btnBatch:hover { background-color: #d35400; }
            QPushButton#btnSaveDir { background-color: #8e44ad; }
            QPushButton#btnSaveDir:hover { background-color: #7d3c98; }
            QGroupBox {
                font-weight: bold; color: #2c3e50;
                border: 1px solid #bdc3c7; border-radius: 5px;
                margin-top: 10px; padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QTextEdit { border: 1px solid #bdc3c7; border-radius: 4px; }
            QProgressBar {
                border: 1px solid #bdc3c7; border-radius: 4px;
                text-align: center; height: 20px;
            }
            QProgressBar::chunk { background-color: #3498db; border-radius: 3px; }
        """)
        self.btn_segment.setObjectName("btnSegment")
        self.btn_batch.setObjectName("btnBatch")
        self.btn_save_dir.setObjectName("btnSaveDir")

    # ---- 按钮事件 ----
    def open_image(self):
        """打开单张图片"""
        ext_filter = "图片文件 (" + " ".join(f"*{e}" for e in VALID_EXTENSIONS) + ")"
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", ext_filter)
        if not path:
            return
        self.current_image_path = path
        self.current_colored_mask = None
        self.display_original(path)
        self._reset_result_label()
        self.txt_info.clear()
        self.lbl_status.setText(f"已加载: {os.path.basename(path)}")
        self.lbl_count.setText("")

    def open_folder(self):
        """打开文件夹，加载所有图片路径（不显示图片，等批量分割时再显示）"""
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        self.folder_image_paths = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(VALID_EXTENSIONS)
        )
        if not self.folder_image_paths:
            QMessageBox.information(self, "提示", "所选文件夹中没有图片")
            return
        self.lbl_status.setText(f"已加载文件夹: {len(self.folder_image_paths)} 张图片")
        self.lbl_count.setText(f"文件夹: {os.path.basename(folder)} ({len(self.folder_image_paths)} 张)")
        # 选择文件夹时不显示图片，等批量分割时再显示
        self.current_image_path = None
        self.current_colored_mask = None
        self._reset_all_labels()

    def select_save_dir(self):
        """选择结果保存路径"""
        path = QFileDialog.getExistingDirectory(self, "选择结果保存路径")
        if path:
            self.save_dir = path
            self.lbl_save_path.setText(f"保存路径: {path}")

    def segment_image(self):
        """单张图片分割，结果保存到指定路径"""
        if not self.current_image_path:
            QMessageBox.information(self, "提示", "请先打开图片")
            return
        if not self.backend:
            QMessageBox.critical(self, "错误", "模型未加载")
            return
        if not self.save_dir:
            QMessageBox.information(self, "提示", "请先选择保存路径")
            return
        self._start_worker([self.current_image_path], single=True)

    def batch_segment(self):
        """批量分割文件夹中的所有图片并保存到指定路径"""
        if not self.folder_image_paths:
            QMessageBox.information(self, "提示", "请先打开文件夹")
            return
        if not self.backend:
            QMessageBox.critical(self, "错误", "模型未加载")
            return
        if not self.save_dir:
            QMessageBox.information(self, "提示", "请先选择保存路径")
            return
        self.batch_results = []
        self._start_worker(self.folder_image_paths, single=False)

    def _start_worker(self, image_paths, single):
        """启动推理工作线程"""
        self._set_buttons_enabled(False)
        self.lbl_status.setText("正在分割..." if single else "批量分割中...")
        self.progress_bar.setVisible(not single)
        if not single:
            self.progress_bar.setValue(0)

        self.worker = PredictWorker(self.backend, image_paths)
        if single:
            self.worker.result_ready.connect(self.on_single_result)
        else:
            self.worker.result_ready.connect(self.on_batch_result)
            self.worker.progress.connect(
                lambda cur, total: self.progress_bar.setValue(int(cur / total * 100))
            )
            self.worker.finished_all.connect(self.on_batch_finished)
        self.worker.finished_all.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()

    # ---- 回调 ----
    def on_single_result(self, result):
        """单张分割完成：保存结果、显示彩色结果与统计信息"""
        # 保存到指定路径
        SegmentationBackend.save_colored(
            result['colored_mask'], os.path.join(self.save_dir, result['image_name'])
        )
        self.current_colored_mask = result['colored_mask']
        self.display_result(result['colored_mask'])
        self.display_info(result)
        self.lbl_status.setText(f"分割完成: {result['image_name']} (已保存)")

    def on_batch_result(self, result):
        """批量分割单张完成：保存结果并更新预览（左侧原图，右侧分割图）"""
        name = result['image_name']
        SegmentationBackend.save_colored(
            result['colored_mask'], os.path.join(self.save_dir, name)
        )
        self.batch_results.append(result)
        # 更新预览显示：先显示原图，再显示分割结果
        self.current_image_path = result['image_path']
        self.current_colored_mask = np.ascontiguousarray(result['colored_mask'])
        self.display_original(result['image_path'])
        self.display_result(result['colored_mask'])
        self.lbl_status.setText(f"已处理: {name}")

    def on_batch_finished(self):
        """批量分割全部完成"""
        n = len(self.batch_results)
        if self.batch_results:
            self.display_info(self.batch_results[-1])
        self.lbl_status.setText(f"批量分割完成: 共 {n} 张，已保存至 {self.save_dir}")
        self.progress_bar.setVisible(False)

    def on_worker_finished(self):
        """工作线程结束：恢复按钮"""
        self._set_buttons_enabled(True)

    def on_worker_error(self, msg):
        """推理错误"""
        self.lbl_status.setText(f"错误: {msg}")

    # ---- 显示辅助 ----
    def display_original(self, image_path):
        """显示原始图片（按Label尺寸等比缩放）"""
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
        self.lbl_original.setPixmap(
            pixmap.scaled(self.lbl_original.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.lbl_original.setStyleSheet("border: 1px solid #bdc3c7; border-radius: 4px;")

    def display_result(self, colored_mask):
        """显示彩色分割结果（RGB numpy -> QPixmap）"""
        self.current_colored_mask = np.ascontiguousarray(colored_mask)
        h, w, _ = self.current_colored_mask.shape
        qimg = QImage(self.current_colored_mask.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.lbl_result.setPixmap(
            pixmap.scaled(self.lbl_result.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.lbl_result.setStyleSheet("border: 1px solid #bdc3c7; border-radius: 4px;")

    def display_info(self, result):
        """显示分割统计信息（推理耗时、类别、面积占比）"""
        stats = result['stats']
        html = f"<b>图片:</b> {result['image_name']}<br>"
        html += f"<b>推理耗时:</b> {result['infer_time']:.1f} ms<br>"
        found = [NAME_MAP[k] for k, v in stats.items() if v['count'] > 0]
        html += f"<b>分割类别:</b> {', '.join(found) if found else '无'}<br>"
        html += "<b>各类别面积占比:</b><br><table cellpadding='3'>"
        for cls_id in sorted(stats.keys()):
            if stats[cls_id]['count'] > 0:
                bgr = COLOR_MAP[cls_id]
                hex_c = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"  # BGR -> RGB hex
                html += (
                    f"<tr><td style='background-color:{hex_c};'>&nbsp;&nbsp;&nbsp;</td>"
                    f"<td>{NAME_MAP[cls_id]}</td>"
                    f"<td>{stats[cls_id]['ratio']:.2f}%</td></tr>"
                )
        html += "</table>"
        self.txt_info.setHtml(html)

    def _reset_result_label(self):
        """重置右侧结果Label为灰色占位"""
        self.lbl_result.clear()
        self.lbl_result.setText("未分割")
        self.lbl_result.setStyleSheet(
            "background-color: #d5d8dc; color: #7f8c8d; "
            "font-size: 14px; border: 1px solid #bdc3c7; border-radius: 4px;"
        )

    def _reset_all_labels(self):
        """重置所有图片显示区域为灰色占位"""
        for lbl in (self.lbl_original, self.lbl_result):
            lbl.clear()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "background-color: #d5d8dc; color: #7f8c8d; "
                "font-size: 14px; border: 1px solid #bdc3c7; border-radius: 4px;"
            )
        self.lbl_original.setText("未加载图片")
        self.lbl_result.setText("未分割")

    def _set_buttons_enabled(self, enabled):
        """切换按钮可用状态"""
        for btn in (self.btn_open_image, self.btn_open_folder, self.btn_segment,
                     self.btn_batch, self.btn_save_dir):
            btn.setEnabled(enabled)

    def resizeEvent(self, event):
        """窗口大小变化时重新缩放图片"""
        if self.current_image_path:
            pixmap = QPixmap(self.current_image_path)
            self.lbl_original.setPixmap(
                pixmap.scaled(self.lbl_original.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        if self.current_colored_mask is not None:
            h, w, _ = self.current_colored_mask.shape
            qimg = QImage(self.current_colored_mask.data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.lbl_result.setPixmap(
                pixmap.scaled(self.lbl_result.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        super().resizeEvent(event)

    def closeEvent(self, event):
        """关闭窗口时清理工作线程"""
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait(3000)
        event.accept()


# ==================== 入口 ====================
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
