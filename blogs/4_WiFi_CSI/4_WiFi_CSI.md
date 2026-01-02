## What is CSI?

CSI (channel state information) provides physical channel measurements in subcarrier-level granularity, and it can be easily accessed from the commodity Wi-Fi network interface controller (NIC).

Most research paper treat the CFR (channel frequency response) sampled at different subcarriers as the CSI data.

CFR is essentially the Fourier transform of CIR (channel impulse response).

## From Raw Data to Statistical Insight

Think of the CSI as a high-resolution snapshot of the channel's frequency response over a short time. A single CSI measurement tells you about the state *at that instant*, but it doesn't reveal the *dynamics*.

The **Autocorrelation Function** transforms this series of snapshots into a tool that answers critical questions about how the channel evolves over time and frequency. It quantifies the "memory" of the channel.

### Key Statistical Properties Revealed by the ACF

#### 1. Temporal Correlation (Over Time)

This is the most common application. By computing the ACF along the time dimension for a single subcarrier, you can determine:

- **Coherence Time 相干时间 ($T_c$):** This is the *maximum time duration* over which the channel's impulse response is essentially invariant. It's a direct measure of how fast the channel is changing.
  - **How to find it:** The ACF of a static channel will be high for short time lags (对于静态信道，时滞越短，信道表现出的相关性越强). The **coherence time is typically defined as the time lag where the ACF magnitude falls below a certain threshold (e.g., 0.5 or 1/e)**. A slow decay indicates a highly correlated (slow-changing) channel, common in static environments. A fast decay indicates a rapidly changing channel, caused by high mobility or Doppler spread.
- **Doppler Spread ($B_d$):** The Doppler spread is the frequency domain dual of coherence time ($T_c ≈ 1/B_d$). It represents the spectral broadening due to motion. The ACF in time is directly related to the Doppler Power Spectral Density via the Fourier Transform. Analyzing the width of the ACF gives you the Doppler spread.

#### 2. Frequency Correlation (Over Subcarriers)

By computing the ACF across different subcarriers at the same time instant, you can determine:

- **Coherence Bandwidth ($B_c$):** This is the *frequency range* over which the channel can be considered "flat" (i.e., the channel's gain and phase are highly correlated).
  - **How to find it:** Similar to coherence time, you find the **frequency lag where the ACF magnitude falls below a threshold**. A slow decay (wide ACF) means a large coherence bandwidth, typical of environments with little multipath (e.g., a line-of-sight link). A fast decay (narrow ACF) means a small coherence bandwidth, indicative of significant delay spread caused by rich multipath.
- **Delay Spread ($σ_τ$):** The delay spread is the time-domain dual of coherence bandwidth ($B_c ≈ 1/σ_τ$). It measures the dispersion of the signal in time due to multiple propagation paths. The ACF in frequency is the Fourier Transform of the Power Delay Profile. The width of this ACF reveals the delay spread.

#### 3. Spatial Correlation (Over Multiple Antennas)

In MIMO systems with multiple antennas, you can compute the spatial ACF across antennas.

- **Coherence Distance ($D_c$):** This is the spatial separation required for two antenna signals to become uncorrelated.
  - **How to find it:** The ACF is calculated over the antenna index. The **spatial lag where the ACF decays** gives the coherence distance. This is crucial for understanding the effectiveness of spatial diversity and multiplexing in MIMO.

## Infer the Environment

By monitoring how the ACF changes, you can infer the environment.

- **Human Activity Recognition:** A person walking through an environment will change the multipath profile. This will directly affect the **delay spread** (changing the frequency ACF) and create Doppler shifts (changing the time ACF). The ACF provides a clean, statistical signature of these activities, which is more robust than looking at raw CSI amplitude/phase.
- **Speed Estimation:** The Doppler spread (derived from the temporal ACF) is directly proportional to the velocity of a moving object. By tracking how the ACF width changes over time, you can estimate a person's or vehicle's speed.
- **Fall Detection / Intrusion Detection:** A sudden, dramatic change in the temporal or frequency statistics (a "break" in the correlation) can be a very reliable indicator of an anomalous event.

## References

1. **Understanding CSI:** https://tns.thss.tsinghua.edu.cn/wst/docs/pre/
