
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors




data = np.array([
    # q0            q1             q2           q3              q4          q5              q6          q7          q8              q9
    [+0.014,np.nan,+0.012,np.nan,+0.016,np.nan,+0.017,np.nan,+0.022,+0.102,np.nan,-0.093,-0.043,-0.030,-0.021,np.nan,-0.012,np.nan,-0.007],
    [+0.011,np.nan,+0.009,np.nan,+0.012,np.nan,+0.012,np.nan,+0.015,+0.035,+0.065,+0.110,np.nan,-0.128,-0.048,np.nan,np.nan,np.nan,-0.008],
    [+0.008,np.nan,+0.008,np.nan,+0.011,np.nan,+0.009,np.nan,+0.014,+0.018,+0.010,+0.029,+0.044,+0.074,np.nan,np.nan,-0.043,np.nan,-0.018],
    [+0.007,np.nan,+0.007,np.nan,+0.008,np.nan,+0.008,np.nan,+0.013,+0.011,+0.013,+0.016,+0.019,+0.025,+0.044,np.nan,np.nan,np.nan,-0.033],
    ])
# Choose a colormap
data = data*100

fig, ax = plt.subplots()

cmap = plt.cm.viridis  # You can choose any colormap

# Set the color for NaN values
cmap = cmap.copy()  # Make a copy to modify
cmap.set_bad(color='gray')  # Set NaN color to gray

# Plot the data
plt.imshow(data, cmap="RdBu",vmin=-0.15, vmax=0.15)
plt.colorbar(label='Value')  # Add a colorbar
# plt.title("2D Colormap with NaN Elements")



def plot_shift_crosstalk( data, pos_q, q_name, ax:plt.Axes, color="red", common_label=None ):
    source_num = data.shape[1]
    detect_num = data.shape[0]
    dx = np.arange(source_num)
    print(common_label)
    for i in range(detect_num):
        if common_label == None:
            ax.plot( dx -pos_q[i], data[i], "o",color=color, label=q_name[i] )
        else:
            if i==detect_num-1:
                ax.plot( dx -pos_q[i], data[i], "o",color=color, label=common_label )
            else:
                ax.plot( dx -pos_q[i], data[i], "o",color=color )

fig_1d, ax = plt.subplots()
pos_q = [10,12,14,16]
q_name = ["q5", "q6", "q7", "q8"]
# Create a 2D array with some NaN values

plot_shift_crosstalk(np.abs(data),pos_q,q_name,ax,"red","20Q19C w/o airbridge (20Q19C_1113ISB_2)")



data = np.array([
    # q0            q1             q2           q3              q4          q5              q6          q7          q8              q9
    [+0.045,+0.050,np.nan,+0.180,np.nan,-0.460,-0.045,-0.042,-0.026],
    [+0.036,+0.025,np.nan,+0.021,+0.045,+0.400,np.nan,-0.180,-0.070],
    [+0.010,+0.006,np.nan,+0.006,+0.012,+0.042,+0.020,+0.120,np.nan],
    ])
pos_q = [4,6,8]
q_name = [ "q2", "q3", "q4"]
plot_shift_crosstalk(np.abs(data*100),pos_q,q_name,ax,"blue","5Q4C airbridge(5Q4C_OS0516AB_9)")


data = np.array([
    # q0            q1             q2           q3              q4          q5              q6          q7          q8              q9
    [np.nan,-0.038,-0.027,-0.020],
    [-0.043,np.nan,-0.056,-0.027],
    [-0.009,-0.029,np.nan,-0.058],
    [+0.003,+0.001,+0.008,np.nan],
    ])
pos_q = [0,1,2,3]
q_name = [ "c2", "q3", "c3", "q4"]
plot_shift_crosstalk(np.abs(data*100),pos_q,q_name,ax,"green","5Q4C Wire bonding(5Q4C_OS0430_7)")


data = np.array([
    # q0            q1             q2           q3              q4          q5              q6          q7          q8              q9
    [-0.02,	    -0.12,	np.nan, 	0.164,	0.049,	0.035],
    [-0.066,	-0.12,	np.nan, 	0.23,	0.049,	np.nan],
    [-0.059,	-0.1,	np.nan, 	np.nan,	0.053,	0.041],
    [-0.052,	np.nan,	np.nan, 	0.152,	0.043,	0.016],
    [-0.057,	-0.091,	np.nan, 	0.059,	0.041,	0.029],

    ])
pos_q = [2, 2, 2, 2, 2]
q_name = [ "q5", "q6", "q7", "q8", "q9"]
plot_shift_crosstalk(np.abs(data*100),pos_q,q_name,ax,"orange","20Q19C(20Q19C_OS241213_4)")


model_x_l = np.linspace(-20,-1,20)
model_x_r = np.linspace(1,20,20)
nearest_crosstalk = 10
ax.plot(model_x_l, -nearest_crosstalk/model_x_l,"--",color="black", linewidth=2, label=f"{nearest_crosstalk}/x")
ax.plot(model_x_r, nearest_crosstalk/model_x_r,"--",color="black", linewidth=2 )
# ax.axhline(y=0.001, color='black', linestyle='--', linewidth=2, label='0')
# ax.axhline(y=-0.001, color='black', linestyle='--', linewidth=2, label='0')
ax.set_yscale('log')

ax.set_xlim(-20,15)
ax.set_ylim(0.1,50)
ax.legend()
plt.show()