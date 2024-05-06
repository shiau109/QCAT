import numpy as np
import matplotlib.pyplot as plt
raw_data = np.load(r'D:\Data\DR2_5Q\1u_Q2Z1_crosstalk_ramsey_20240118_1419.npz', allow_pickle=True)# ["arr_0"].item()
# tomo_data =
other_info = {}
for k, v in raw_data.items():
    print(k, v.shape)
    if k in ["paras","setting"]:
        other_info[k]=v.item()

for k, v in other_info.items():
    print(k)

d_z_target_amp = other_info["paras"]["d_z_target_amp"]
d_z_crosstalk_amp = other_info["paras"]["d_z_crosstalk_amp"]




# Create a 2D numpy array (example data)
data = raw_data["q2_ro"][0]
offset = np.mean(data)
print(offset)
data -= offset



# Apply Gaussian smoothing
from scipy.ndimage import gaussian_filter
sigma = 1.0  # Adjust the standard deviation based on your needs
smoothed_data = gaussian_filter(data, sigma=sigma)


print(data.shape, d_z_target_amp.shape, d_z_crosstalk_amp.shape)
# M, N = data.shape

def get_freq_axes( axes ):
    f_axes = []
    for a in axes:
        ax_len = a.shape[-1]
        d = np.abs(a[0]-a[1])
        print(f"length: {ax_len}, delta:{d}")

        f_axes.append( np.fft.fftshift(np.fft.fftfreq(ax_len, d=d)) )
    return f_axes

from scipy.interpolate import RegularGridInterpolator

def get_weighted_pos( data, axes ):
    max_index = np.argmax(data)
    indices = np.unravel_index(max_index, data.shape)
    
    edge_min_idx = [indices[0]-2, indices[1]-2]
    edge_max_idx = [indices[0]+2, indices[1]+2]

    partial_amp = data[edge_min_idx[0]:edge_max_idx[0]+1,edge_min_idx[1]:edge_max_idx[1]+1]
    f_z_t_range = axes[0][edge_min_idx[0]:edge_max_idx[0]+1]
    f_z_c_range = axes[1][edge_min_idx[1]:edge_max_idx[1]+1]
    f_z_t_range_new = np.linspace(f_z_t_range[0],f_z_t_range[-1],10)
    f_z_c_range_new = np.linspace(f_z_c_range[0],f_z_c_range[-1],10)
    print(partial_amp.shape, f_z_t_range.shape, f_z_c_range.shape)
    print(f_z_c_range)
    interp = RegularGridInterpolator((f_z_t_range, f_z_c_range), partial_amp, method="cubic",
                                     fill_value=None, bounds_error=False)
    print(f_z_t_range, f_z_t_range_new)
    tv, cv = np.meshgrid(f_z_t_range_new, f_z_c_range_new, indexing="ij")
    f_z_target_pos, f_z_crosstalk_pos = get_max_pos(interp((tv, cv)), [f_z_t_range_new, f_z_c_range_new])

    # tv, cv = np.meshgrid(f_z_t_range_new, f_z_c_range_new, indexing="ij")
    # print(f_z_t_range, f_z_c_range)
    # f_z_target_pos = np.average(tv, weights=partial_amp)
    # f_z_crosstalk_pos = np.average(cv, weights=partial_amp)

    # print(f_z_target_pos, f_z_crosstalk_pos)
    return f_z_target_pos, f_z_crosstalk_pos

def get_max_pos( data, axes ):

    max_index = np.argmax(data)
    max_indices = np.unravel_index(max_index, data.shape)
    f_z_target_pos = axes[0][max_indices[0]]
    f_z_crosstalk_pos = axes[1][max_indices[1]]
    print(max_indices)
    print("Maximum value:", data[max_indices])
    print(f"f_z_target: {axes[0][max_indices[0]]}")
    print(f"f_z_crosstalk: {axes[1][max_indices[1]]}")
    return f_z_target_pos, f_z_crosstalk_pos

def get_extend( data, axes:list, extend_num = 50 ):
    extended_data = np.pad(data, extend_num, mode='constant', constant_values=0)
    ext_axes = []
    for a in axes:
        ax_len = a.shape[-1]
        d = a[1]-a[0]
        total_point = ax_len+2*extend_num
        ext_axes.append( np.linspace( a[0]-extend_num*d, a[-1]+extend_num*d, total_point) )
    print(extended_data,extended_data.shape)

    return extended_data, ext_axes


def get_interp( data, axes:list, extend_num = 50 ):
    from scipy.interpolate import interp2d
    # Create an interpolation function
    interp_func = interp2d(axes[1], axes[0], data, kind='linear')
    # Define new points for interpolation
    ext_axes = []
    for a in axes:
        ax_len = a.shape[-1]
        d = a[1]-a[0]
        total_point = ax_len+2*extend_num
        ext_axes.append( np.linspace( a[0], a[-1], total_point) )
    # Interpolate values at new points
    extended_data = interp_func(ext_axes[1], ext_axes[0] )
    print(extended_data,extended_data.shape)

    return extended_data, ext_axes

data, axes = get_extend(data, [d_z_target_amp, d_z_crosstalk_amp], 100)
# data, axes = get_interp(data, [d_z_target_amp, d_z_crosstalk_amp], 100)


f_axes = get_freq_axes( axes )

# Compute the 2D Fourier Transform
fft_result = np.fft.fft2(data)

# Shift zero frequency components to the center
fft_result_shifted = np.fft.fftshift(fft_result)

# Compute the magnitude spectrum (absolute values)
magnitude_spectrum = np.abs(fft_result_shifted)

get_max_pos(magnitude_spectrum, f_axes)
# f_z_target_pos, f_z_crosstalk_pos = get_weighted_pos(magnitude_spectrum, f_axes )
f_z_target_pos, f_z_crosstalk_pos = get_max_pos(magnitude_spectrum, f_axes)
z_slope = f_z_target_pos/f_z_crosstalk_pos

print(f"z_slope: {z_slope}, {-1/z_slope}")
# Display the original image and its Fourier Transform
# plt.figure(figsize=(4, 8))

fig, ax = plt.subplots(ncols=2)
fig.set_size_inches(10, 5)
ax[0].pcolormesh(axes[1]*1000, axes[0]*1000, data, cmap='gray')
ax[0].plot([d_z_crosstalk_amp[0]*1000,d_z_crosstalk_amp[-1]*1000],[-d_z_crosstalk_amp[0]/z_slope*1000,-d_z_crosstalk_amp[-1]/z_slope*1000],color="r",linewidth=5)
ax[0].set_title('Original Image')
ax[0].set_xlabel(f"Q1 Delta Voltage (mV)")
ax[0].set_ylabel(f"Q2 Delta Voltage (mV)")
# plt.pcolormesh(f_z_crosstalk, f_z_target, np.log1p(magnitude_spectrum), cmap='gray')  # Use log scale for better visualization
ax[1].pcolormesh(f_axes[1]/1000, f_axes[0]/1000, magnitude_spectrum, cmap='gray')  # Use log scale for better visualization
ax[1].plot([-f_z_crosstalk_pos/1000,f_z_crosstalk_pos/1000],[-f_z_target_pos/1000,f_z_target_pos/1000],"o",color="r",markersize=5)
ax[1].plot([-f_z_crosstalk_pos/1000,f_z_crosstalk_pos/1000],[-f_z_target_pos/1000,f_z_target_pos/1000],color="r",linewidth=1)
ax[1].set_xlabel(f"Q1 wavenumber (1/mV)")
ax[1].set_ylabel(f"Q2 wavenumber (1/mV)")
ax[1].set_title('2D Fourier Transform (Magnitude Spectrum)')

plt.show()
