import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks, savgol_filter
from scipy.fft import fft
import matplotlib.pyplot as plt
from collections import defaultdict


class IntegratedRadarActionCounter:
    def __init__(self):
        # 雷达参数
        self.n_samples = 512
        self.n_chirps = 16
        self.n_rx = 4
        self.n_frames = 10

        # 动作参数映射
        self.action_params = {
            '交替腿举': {'min_period': 0.6, 'max_period': 2.0, 'prominence': 0.15},
            '仰卧起坐': {'min_period': 0.5, 'max_period': 2.0, 'prominence': 0.15},
            '俄罗斯转体': {'min_period': 0.6, 'max_period': 2.0, 'prominence': 0.15},
            '俯卧撑': {'min_period': 0.5, 'max_period': 2.0, 'prominence': 0.1},
            '前弓步': {'min_period': 0.3, 'max_period': 1.2, 'prominence': 0.05},
            '开合跳': {'min_period': 0.2, 'max_period': 0.8, 'prominence': 0.05},
            '深蹲': {'min_period': 0.5, 'max_period': 2.0, 'prominence': 0.1},
            '登山者': {'min_period': 0.4, 'max_period': 1.5, 'prominence': 0.05},
            '高抬腿': {'min_period': 0.15, 'max_period': 0.6, 'prominence': 0.05},
            '鸟狗式': {'min_period': 0.8, 'max_period': 3.0, 'prominence': 0.15}
        }

        # 类别ID到动作名称的映射
        self.id_to_action = {
            0: "交替腿举", 1: "交替腿举",
            2: "仰卧起坐", 3: "仰卧起坐",
            4: "俄罗斯转体", 5: "俄罗斯转体",
            6: "俯卧撑", 7: "俯卧撑",
            8: "前弓步", 9: "前弓步",
            10: "开合跳", 11: "开合跳",
            12: "深蹲", 13: "深蹲",
            14: "登山者", 15: "登山者",
            16: "高抬腿", 17: "高抬腿",
            18: "鸟狗式", 19: "鸟狗式"
        }

        # 统计信息
        self.total_counts = defaultdict(int)
        self.session_counts = defaultdict(int)

    def load_radar_data(self, filepath):
        """加载雷达数据"""
        try:
            data = sio.loadmat(filepath)
            possible_keys = ['adcSampleAll', 'data', 'radar_data', 'adc_data']

            radar_data = None
            for key in possible_keys:
                if key in data:
                    radar_data = data[key]
                    break

            if radar_data is None:
                for key, value in data.items():
                    if not key.startswith('__') and isinstance(value, np.ndarray):
                        if value.ndim == 4:
                            radar_data = value
                            break

            if radar_data is None:
                raise ValueError("Cannot find radar data in the .mat file")

            print(f"Loaded data shape: {radar_data.shape}")
            return radar_data

        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            raise

    def range_doppler_processing(self, frame_data):
        """距离-多普勒处理"""
        # Range FFT
        range_fft = fft(frame_data, axis=0)
        range_fft = range_fft[:self.n_samples // 2, :, :]

        # Doppler FFT
        doppler_fft = fft(range_fft, axis=1)
        doppler_fft = np.fft.fftshift(doppler_fft, axes=1)

        # 计算功率谱
        rd_map = np.sum(np.abs(doppler_fft) ** 2, axis=2)

        return rd_map

    def extract_motion_features(self, radar_data):
        """提取运动特征"""
        n_frames = radar_data.shape[3]
        features = []

        # 初始化range-time map
        range_time_map = np.zeros((self.n_samples // 2, n_frames))

        for i in range(n_frames):
            frame = radar_data[:, :, :, i]
            rd_map = self.range_doppler_processing(frame)

            # 排除静态杂波（中心多普勒频率）
            center_bin = self.n_chirps // 2
            doppler_mask = np.ones(self.n_chirps, dtype=bool)
            doppler_mask[center_bin - 1:center_bin + 2] = False

            # 计算运动能量
            doppler_energy = np.sum(rd_map[:, doppler_mask], axis=0)

            # Range profile
            range_bins = slice(5, 50)
            range_energy = np.sum(rd_map[range_bins, :], axis=1)

            # 总运动强度
            motion_intensity = np.sum(rd_map[range_bins, doppler_mask])

            # 最大多普勒
            max_doppler = np.max(np.abs(doppler_energy))

            # 构建range-time map
            range_time_map[:, i] = np.sum(rd_map[:, doppler_mask], axis=1)

            features.append({
                'doppler_energy': doppler_energy,
                'range_energy': range_energy,
                'motion_intensity': motion_intensity,
                'max_doppler': max_doppler,
                'rd_map': rd_map
            })

        # 添加range-time map到features
        features.append({'range_time_map': range_time_map})

        return features

    def smooth_signal(self, signal):
        """平滑信号"""
        if len(signal) < 5:
            return signal

        window_length = min(5, len(signal))
        if window_length % 2 == 0:
            window_length -= 1

        if window_length >= 3:
            return savgol_filter(signal, window_length, 2)
        else:
            return signal

    def analyze_temporal_pattern(self, features, action_type, distance_m=2):
        """分析时间模式并计数"""
        range_time_map = features[-1]['range_time_map']

        # 获取动作参数
        params = self.action_params.get(action_type, {
            'min_period': 0.5, 'max_period': 3.0, 'prominence': 0.2
        })

        # 距离分辨率
        range_resolution = 0.088
        center_bin = int(distance_m / range_resolution)

        # 选择分析窗口
        if distance_m <= 2:
            window_size = 12
        else:
            window_size = 18

        start_bin = max(0, center_bin - window_size // 2)
        end_bin = min(range_time_map.shape[0], start_bin + window_size)

        # 提取运动曲线
        motion_curve = np.sum(range_time_map[start_bin:end_bin, :], axis=0)

        # 归一化
        if np.max(motion_curve) > 0:
            motion_curve_norm = motion_curve / np.max(motion_curve)
        else:
            return 0, motion_curve, []

        # 准备多种信号进行分析
        curves_to_analyze = []

        # 1. 原始信号
        curves_to_analyze.append(('raw', motion_curve_norm))

        # 2. 轻度平滑信号
        if len(motion_curve_norm) >= 3:
            light_smooth = self.smooth_signal(motion_curve_norm)
            curves_to_analyze.append(('light', light_smooth))

        # 3. 差分信号（用于检测快速变化）
        if len(motion_curve_norm) > 1:
            diff_signal = np.diff(motion_curve_norm)
            diff_signal = np.concatenate([[0], diff_signal])
            curves_to_analyze.append(('diff', np.abs(diff_signal)))

        # 收集所有检测到的峰值
        all_peaks = []

        for signal_type, signal in curves_to_analyze:
            if signal_type == 'diff':
                adj_prominence = params['prominence'] * 0.15
                min_dist = 1
            else:
                adj_prominence = params['prominence'] * 0.1
                frame_duration = 0.36
                min_dist = max(1, int(params['min_period'] / frame_duration * 0.5))

            peaks, properties = find_peaks(
                signal,
                prominence=adj_prominence,
                distance=min_dist,
                height=0.1
            )

            # 对于差分信号，也检测负峰
            if signal_type == 'diff' and len(peaks) < 5:
                neg_peaks, _ = find_peaks(
                    -signal,
                    prominence=adj_prominence,
                    distance=min_dist
                )
                peaks = np.sort(np.concatenate([peaks, neg_peaks]))

            all_peaks.extend(peaks)

        # 去重和过滤峰值
        if len(all_peaks) > 0:
            all_peaks = np.array(all_peaks)
            all_peaks = np.unique(all_peaks)

            # 过滤太近的峰值
            filtered_peaks = []
            min_separation = max(1, int(params['min_period'] / 0.36 * 0.3))

            for peak in sorted(all_peaks):
                if len(filtered_peaks) == 0 or peak - filtered_peaks[-1] >= min_separation:
                    filtered_peaks.append(peak)

            peaks = np.array(filtered_peaks)
        else:
            peaks = np.array([])

        # 如果峰值太少，尝试其他方法
        if len(peaks) < 3:
            # 阈值法
            threshold = np.mean(motion_curve_norm) + 0.3 * np.std(motion_curve_norm)
            above_threshold = motion_curve_norm > threshold

            regions = []
            in_region = False
            start_idx = 0

            for i in range(len(above_threshold)):
                if above_threshold[i] and not in_region:
                    start_idx = i
                    in_region = True
                elif not above_threshold[i] and in_region:
                    if i - start_idx >= 1:
                        regions.append((start_idx, i))
                    in_region = False

            if in_region:
                regions.append((start_idx, len(above_threshold)))

            if len(regions) > len(peaks):
                peaks = np.array([(r[0] + r[1]) // 2 for r in regions])

        # 动作计数
        action_count = len(peaks)

        return action_count, motion_curve_norm, peaks

    def process_file(self, filepath, prediction_id):
        """处理文件并根据预测ID进行计数"""
        # 获取动作类型
        action_name = self.id_to_action.get(prediction_id, None)

        # 判断是否为标准动作
        is_standard = (prediction_id % 2 == 1)

        if not is_standard or action_name is None:
            return {
                'action': action_name,
                'is_standard': False,
                'current_count': 0,
                'total_count': self.total_counts.get(action_name, 0) if action_name else 0,
                'new_action_detected': False,
                'prediction': prediction_id,
            }

        try:
            # 加载雷达数据
            radar_data = self.load_radar_data(filepath)

            if radar_data.shape[3] < 2:
                print(f"Warning: Not enough frames in {filepath}")
                return {
                    'action': action_name,
                    'is_standard': True,
                    'current_count': 0,
                    'total_count': self.total_counts[action_name],
                    'new_action_detected': False,
                    'prediction': prediction_id,
                }

            # 提取特征
            features = self.extract_motion_features(radar_data)

            # 分析并计数
            count, motion_curve, peaks = self.analyze_temporal_pattern(
                features, action_name, distance_m=2
            )

            # 更新统计
            self.session_counts[action_name] = count
            self.total_counts[action_name] += count

            # 计算平均运动强度
            motion_intensities = [f['motion_intensity'] for f in features[:-1]]
            avg_motion_intensity = np.mean(motion_intensities) if motion_intensities else 0

            print(f"Detected {count} actions for {action_name}")

            # 判断是否检测到新动作
            new_action_detected = count > 0

            return {
                'action': action_name,
                'is_standard': True,
                'current_count': count,
                'total_count': self.total_counts[action_name],
                'prediction': prediction_id,
                #'new_action_detected': new_action_detected,
                #'session_count': self.session_counts[action_name],
                #'motion_intensity': avg_motion_intensity,
                #'motion_curve': motion_curve,
                #'peaks': peaks
            }

        except Exception as e:
            print(f"Error processing file: {e}")
            return {
                'action': action_name,
                'is_standard': is_standard,
                'current_count': 0,
                'total_count': self.total_counts.get(action_name, 0) if action_name else 0,
                'new_action_detected': False,
                'prediction': prediction_id,
                'error': str(e)
            }

    def visualize_results(self, result, features=None):
        """可视化结果"""
        if 'motion_curve' not in result or 'peaks' not in result:
            print("No visualization data available")
            return

        motion_curve = result['motion_curve']
        peaks = result['peaks']
        action_type = result['action']

        plt.figure(figsize=(12, 6))
        time_axis = np.arange(len(motion_curve)) * 0.36
        plt.plot(time_axis, motion_curve, 'b-', linewidth=2, label='Motion Intensity')
        plt.plot(time_axis[peaks], motion_curve[peaks], 'ro',
                 markersize=10, label=f'Detected Actions (n={len(peaks)})')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Normalized Motion Intensity')
        plt.title(f'Action Counting for {action_type}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def get_statistics(self):
        """获取统计信息"""
        return {
            'total_counts': dict(self.total_counts),
            'session_counts': dict(self.session_counts)
        }

    def reset(self, reset_type='all'):
        """重置计数器"""
        if reset_type == 'all':
            self.total_counts.clear()
            self.session_counts.clear()
        elif reset_type == 'session':
            self.session_counts.clear()


# 全局计数器实例
action_counter = IntegratedRadarActionCounter()


def get_action_count_result(mat_filepath, prediction_result):
    """供外部调用的接口函数"""
    return action_counter.process_file(mat_filepath, prediction_result)


def get_counter_statistics():
    """获取计数器统计信息"""
    return action_counter.get_statistics()


def reset_counter(reset_type='all'):
    """重置计数器"""
    action_counter.reset(reset_type)