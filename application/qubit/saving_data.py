#If you used the jupyter notebooks to analyze data, create another cell and paste the following code to save the data and figures

from qcat.utility.save_data import save_nc, save_fig, create_folder


save_dir = 0 #put the project directory here, the code will create subfolders
folder_label = 0 #name your folder here
save_name = 0 #Put the analysis name here
figure_name = 0 
folder_save_dir = create_folder(save_dir, folder_label)
save_nc(folder_save_dir, save_name, dataset)
save_fig(folder_save_dir, figure_name)