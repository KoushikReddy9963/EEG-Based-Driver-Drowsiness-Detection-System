import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class InterpretableCNN(nn.Module):
    def __init__(self):
        super(InterpretableCNN, self).__init__()
        self.pointwise = nn.Conv1d(30, 16, kernel_size=1)
        self.depthwise = nn.Conv1d(16, 32, kernel_size=64, groups=16, padding=31)
        self.activ = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(32)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pointwise(x)
        x = self.depthwise(x)
        x = self.activ(x)
        x = self.batchnorm(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x

class VisTech:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def heatmap_calculation(self, batchInput, sampleidx, state, radius=32):
        batchActiv1 = self.model.pointwise(batchInput)
        batchActiv2 = self.model.depthwise(batchActiv1)
        batchActiv3 = self.model.activ(batchActiv2)
        batchActiv4 = self.model.batchnorm(batchActiv3)

        layer1weights = self.model.pointwise.weight.detach().cpu().numpy().squeeze()
        layer2weights = self.model.depthwise.weight.detach().cpu().numpy().squeeze()
        layer6weights = self.model.fc.weight.detach().cpu().numpy().squeeze()

        sampleInput = batchInput[sampleidx].detach().cpu().numpy().squeeze()
        sampleActiv2 = batchActiv2[sampleidx].detach().cpu().numpy().squeeze()
        sampleActiv4 = batchActiv4[sampleidx].detach().cpu().numpy().squeeze()

        sampleChannel, sampleLength = sampleInput.shape
        kernelLength = self.model.depthwise.kernel_size[0]
        p = self.model.depthwise.padding[0]
        padded_input = np.pad(sampleInput, ((0,0),(p,p)), 'constant')

        CAM = sampleActiv4 * layer6weights[state, :, np.newaxis]
        CAMthres = np.sort(CAM, axis=None)[-100] if len(CAM.flatten()) >= 100 else CAM.min()
        fixationmap = CAM > CAMthres
        valid_sign = sampleActiv4 * sampleActiv2 > 0
        fixationmap = fixationmap & valid_sign

        fixationmap0 = np.zeros((sampleChannel, sampleLength))
        for i in range(fixationmap.shape[0]):
            for j in range(fixationmap.shape[1]):
                if fixationmap[i, j]:
                    channel_idx = i // 2
                    temporal_weights = layer2weights[i]
                    spatial_weights = layer1weights[channel_idx]
                    contribution = np.sum(padded_input[:, j:j + kernelLength] * temporal_weights, axis=1) * spatial_weights
                    p = np.argmax(contribution)
                    q = j
                    if q < sampleLength:
                        fixationmap0[p, q] = 1

        heatmap = np.zeros((sampleChannel, sampleLength))
        x = np.arange(sampleLength)
        for p in range(sampleChannel):
            for q in range(sampleLength):
                if fixationmap0[p, q]:
                    heatmap[p] += np.exp(-(x - q) ** 2 / (2 * radius ** 2)) / (radius * np.sqrt(2 * np.pi))
        heatmap = (heatmap - np.mean(heatmap)) / np.std(heatmap)
        return heatmap

    def generate_heatmap(self, batchInput, sampleidx, subid, samplelabel, likelihood):
        state = 0 if likelihood[0] > likelihood[1] else 1

        sampleInput = batchInput[sampleidx].cpu().numpy().squeeze()
        sampleChannel, sampleLength = sampleInput.shape
        heatmap = self.heatmap_calculation(batchInput, sampleidx, state)

        # Figure 1: EEG Signals with Activation Heatmap
        fig1 = plt.figure(figsize=(12, 10))
        ax1 = fig1.add_subplot(111)
        xx = np.arange(sampleLength)
        channel_spacing = np.percentile(sampleInput, 99) * 2

        # Global normalization for consistent color scaling across channels
        heatmap_min = np.min(heatmap)
        heatmap_max = np.max(heatmap)

        for ch in range(sampleChannel):
            y = sampleInput[ch] + (sampleChannel - ch) * channel_spacing
            segments = self._create_segments(xx, y)
            array = (heatmap[ch, :-1] + heatmap[ch, 1:]) / 2
            lc = LineCollection(segments, cmap='viridis', array=array, 
                               norm=plt.Normalize(heatmap_min, heatmap_max), linewidth=1.5)
            ax1.add_collection(lc)

        ax1.set_xlim(0, sampleLength)
        ax1.set_ylim(-channel_spacing, sampleChannel * channel_spacing)
        channelnames = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz',
                        'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz',
                        'CP4', 'TP8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']
        ax1.set_yticks([(sampleChannel - i) * channel_spacing for i in range(sampleChannel)])
        ax1.set_yticklabels(channelnames[::-1])
        ax1.set_xlabel("Time Samples")
        ax1.set_title(f"Figure 1: EEG Signals with Activation Heatmap\nSubject {subid} ")

        # Add colorbar to indicate important regions for CNN
        cbar = fig1.colorbar(lc, ax=ax1)
        cbar.set_label("Activation Importance")

        # Figure 2: Spectral Power Distribution
        # fig2 = plt.figure(figsize=(8, 10))
        # ax2 = fig2.add_subplot(111)
        # freqs = np.fft.rfftfreq(sampleLength, 1 / 128)
        # psd = np.abs(np.fft.rfft(sampleInput, axis=1)) ** 2
        # bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}
        # for band_name, (f_low, f_high) in bands.items():
        #     mask = (freqs >= f_low) & (freqs <= f_high)
        #     band_power = np.mean(psd[:, mask], axis=1)
        #     ax2.plot(band_power, label=band_name)
        
        # ax2.legend(title="Frequency Bands")
        # ax2.set_xlabel("Channel")
        # ax2.set_ylabel("Power")
        # ax2.set_xticks(range(sampleChannel))
        # ax2.set_xticklabels(channelnames[::-1], rotation=90)
        # ax2.set_title(f"Figure 2: Spectral Power Distribution\nSubject {subid}")

        # Adjust layout and display both figures
        fig1.tight_layout()
        # fig2.tight_layout()
        plt.show()

    def _create_segments(self, x, y):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        return np.concatenate([points[:-1], points[1:]], axis=1)

if __name__ == "__main__":
    model = InterpretableCNN()
    vis = VisTech(model)
    synthetic_eeg = torch.randn(1, 30, 384)
    vis.generate_heatmap(synthetic_eeg, 0, 1, 0, [0.7, 0.3])