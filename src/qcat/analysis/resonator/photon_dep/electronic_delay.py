from scipy import stats
import numpy as np


def get_myDelay( fdata, zdata, edge_ratio = 0.1 ):
    total_length = fdata.shape[-1]
    edge_range = int(len(fdata)*edge_ratio)

    phase2 = np.unwrap(np.angle(zdata))
    start_delay = section_delay( fdata, zdata, start=0, end=edge_range)
    end_delay = section_delay( fdata, zdata, start=total_length-edge_range, end=total_length)

    endpoint_1 = (np.mean(fdata[-edge_range:]), np.mean(phase2[-edge_range:]))
    endpoint_2 = (np.mean(fdata[0:edge_range]), np.mean(phase2[0:edge_range]))
    se_delay = two_point_delay( endpoint_1, endpoint_2 )

    return (start_delay+end_delay+2*se_delay)/4


def section_delay( fdata, zdata, start=0, end=1 ):
    """
    Arg
    start : is the index of position point in total data point
    end : is the index of position point in total data point
    """

    phase2 = np.unwrap(np.angle(zdata))
    gradient, intercept, r_value, p_value, std_err = stats.linregress(fdata[start:end],phase2[start:end])
    delay = gradient*(-1.)/(np.pi*2.)

    return delay

def two_point_delay( first, second ):
    
    if first[0] > second[0]:
        del_f = first[0]-second[0]
        del_phase = first[1]-second[1]
    elif first[0] < second[0]:
        del_f = second[0]-first[0]
        del_phase = second[1]-second[1] 
    gradient = del_phase/del_f
    delay = gradient*(-1.)/(np.pi*2.)
    return delay