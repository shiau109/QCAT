import numpy as np


# # Apply Gaussian smoothing
# from scipy.ndimage import gaussian_filter
# sigma = 1.0  # Adjust the standard deviation based on your needs
# smoothed_data = gaussian_filter(data, sigma=sigma)


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
    # print(partial_amp.shape, f_z_t_range.shape, f_z_c_range.shape)
    interp = RegularGridInterpolator((f_z_t_range, f_z_c_range), partial_amp, method="cubic",
                                     fill_value=None, bounds_error=False)
    # print(f_z_t_range, f_z_t_range_new)
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

def get_fft_mag( data ):
    # Compute the 2D Fourier Transform
    fft_result = np.fft.fft2(data)
    # Shift zero frequency components to the center
    fft_result_shifted = np.fft.fftshift(fft_result)
    # Compute the magnitude spectrum (absolute values)
    magnitude_spectrum = np.abs(fft_result_shifted)
    return magnitude_spectrum

def analysis_crosstalk_value( z1, z2, data ):
    offset = np.mean(data)
    data -= offset
    data, axes = get_extend(data, [z1, z2], 100)
    # data, axes = get_interp(data, [d_z_target_amp, d_z_crosstalk_amp], 100)

    f_axes = get_freq_axes( axes )
    magnitude_spectrum = get_fft_mag(data)
    # f_z_target_pos, f_z_crosstalk_pos = get_weighted_pos(magnitude_spectrum, f_axes )
    f_z_target_pos, f_z_crosstalk_pos = get_max_pos(magnitude_spectrum, f_axes)
    z_slope = f_z_target_pos/f_z_crosstalk_pos
    crosstalk = -1/z_slope
    print(f"k space: {z_slope}")
    print(f"crosstalk: {crosstalk}")

    return crosstalk, f_axes, magnitude_spectrum

