import numpy as np
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt


class RadarPreprocessor:
    def __init__(self, fs=10, lowcut=0.2, highcut=3.5, order=2,
                 cutout_ratio=0.1, shift_range=5, brightness_range=
                 (0.9, 1.1), contrast_range=(0.9, 1.1)):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.cutout_ratio = cutout_ratio
        self.shift_range = shift_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    @staticmethod
    def load_adc_data(filepath):
        """Load raw ADC data from .mat file, shape (512, 16, 4, T)."""
        try:
            mat = sio.loadmat(filepath)
            return mat['adcSampleAll']
        except FileNotFoundError:
            raise FileNotFoundError(f"MAT file not found: {filepath}")
        except KeyError:
            raise KeyError(f"'adcSampleAll' not found in MAT file: "
                           f"{filepath}")

    @staticmethod
    def remove_dc(adc):
        """Remove DC (static clutter) along range axis."""
        dc = np.mean(adc, axis=0, keepdims=True)
        return adc - dc

    @staticmethod
    def compute_rtm(adc_dc):
        """Compute Range-Time Map (RTM) as mean over chirps and antennas."""
        range_fft = np.fft.fft(adc_dc, axis=0)
        range_fft = range_fft[: 256, :, :, :]
        rtm = np.mean(np.abs(range_fft), axis=(1, 2))  # shape: (256, T)
        return rtm

    @staticmethod
    def compute_dtm(adc_dc):
        """Compute Doppler-Time Map (DTM) as mean over range and antennas."""
        range_fft = np.fft.fft(adc_dc, axis=0)
        range_fft = range_fft[:256, :, :, :]
        doppler = np.fft.fft(range_fft, axis=1)
        doppler = np.fft.fftshift(doppler, axes=1)
        dtm = np.mean(np.abs(doppler), axis=(0, 2))  # shape: (16, T)
        return dtm

    def butter_bandpass(self):
        """Design a Butterworth bandpass filter."""
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return b, a

    @staticmethod
    def apply_filter(x, b, a):
        """Apply zero-phase bandpass filter along first axis."""
        orig_shape = x.shape
        flat = x.reshape(x.shape[0], -1)
        filtered = filtfilt(b, a, flat, axis=0)
        return filtered.reshape(orig_shape)

    @staticmethod
    def crop_roi(rtm, dtm):
        """Crop ROI in range and velocity domains."""
        rtm_c = rtm[18: 73, :]  # keep range bins 18-72
        dtm_c = dtm[1: -1, :]  # remove first and last doppler bins
        return rtm_c, dtm_c

    @staticmethod
    def normalize(x):
        """Min-Max normalize to [0,1]."""
        return (x - x.min()) / (x.max() - x.min() + 1e-6)

    @staticmethod
    def resize(x):
        """Resize 2D array x to 224x224 using bilinear interpolation."""
        return cv2.resize(x, (224, 224), interpolation=cv2.INTER_LINEAR)

    def spatial_augment(self, x):
        """Apply random cutout and shift for spatial augmentation."""
        h, w = x.shape

        # random cutout
        ch, cw = int(h * self.cutout_ratio), int(w * self.cutout_ratio)
        x_aug = x.copy()
        y = np.random.randint(0, h - ch)
        x0 = np.random.randint(0, w - cw)
        x_aug[y: y + ch, x0: x0 + cw] = 0

        # random shift
        shift_x = np.random.randint(-self.shift_range, self.shift_range + 1)
        shift_y = np.random.randint(-self.shift_range, self.shift_range + 1)
        x_aug = np.roll(np.roll(x_aug, shift_x, axis=0), shift_y, axis=1)
        return x_aug

    @staticmethod
    def to_pseudo_color(x_gray):
        """Convert to 224x224x3 RGB using jet colormap."""
        cmap = plt.colormaps['jet']
        rgba = cmap(x_gray)  # shape: (224,224,4)
        rgb = np.delete(rgba, 3, axis=2)  # remove alpha -> (224,224,3)
        return (rgb * 255).astype(np.uint8)

    def color_jitter(self, img):
        """Apply random brightness / contrast jitter on RGB image."""
        img = img.astype(np.float32)
        brightness = np.random.uniform(*self.brightness_range)
        contrast = np.random.uniform(*self.contrast_range)
        img = img * contrast + (brightness - 1) * 255
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def preprocess_pipeline(self, filepath, visualize=False):
        """Preprocessing pipeline for each .mat file, returning RTM and
        DTM RGB tensors."""
        # 1. Load data and remove DC
        adc = self.load_adc_data(filepath)
        adc_dc = self.remove_dc(adc)

        # 2. Compute RTM and DTM
        rtm = self.compute_rtm(adc_dc)  # (256, T)
        dtm = self.compute_dtm(adc_dc)  # (16, T)

        # 3. Band pass filtering
        b, a = self.butter_bandpass()
        rtm = self.apply_filter(rtm, b, a)  # shape: (256, T)
        dtm = self.apply_filter(dtm, b, a)  # shape: (16, T)

        # 4. ROI cropping
        rtm_c, dtm_c = self.crop_roi(rtm, dtm)

        # 5. Normalize
        rtm_n = self.normalize(rtm_c)
        dtm_n = self.normalize(dtm_c)

        # 6. Resize to 224x224
        rtm_r = self.resize(rtm_n)
        dtm_r = self.resize(dtm_n)

        # 7. Spatial augmentation
        rtm_s = self.spatial_augment(rtm_r)
        dtm_s = self.spatial_augment(dtm_r)

        # 8. Pseudo-color mapping
        rtm_rgb = self.to_pseudo_color(rtm_s)
        dtm_rgb = self.to_pseudo_color(dtm_s)

        # 9. Color jitter
        rtm_j = self.color_jitter(rtm_rgb)
        dtm_j = self.color_jitter(dtm_rgb)

        # 10. Visualization
        if visualize:
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(rtm_j)
            axs[0].set_title('RTM (RGB)')
            axs[1].imshow(dtm_j)
            axs[1].set_title('DTM (RGB)')
            plt.suptitle(os.path.basename(filepath))
            plt.tight_layout()
            plt.savefig(f"./vis/{os.path.basename(filepath)}.png")
            plt.close()

        return rtm_j, dtm_j

    def preprocess_all_data(self):
        """Full preprocessing pipeline, traversing all files."""
        samples, labels, file_paths = [], [], []
        for dirpath, _, filenames in os.walk('./data'):
            for file in filenames:
                if file.endswith('.mat'):
                    filepath = os.path.join(dirpath, file)
                    try:
                        rtm, dtm = self.preprocess_pipeline(filepath)
                        samples.append((rtm, dtm))
                        file_paths.append(filepath)
                        action = os.path.basename(os.path.dirname(filepath))
                        if 'nstd' in file.lower():
                            label = f"{action}不标准"
                        else:
                            label = f"{action}标准"
                        labels.append(label)
                    except Exception as e:
                        print(f"[ERROR] Failed to process {filepath}: {e}")
        return samples, labels, file_paths
