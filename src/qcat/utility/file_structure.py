from os import listdir,makedirs
from shutil import rmtree
from os.path import isfile, join,exists
import numpy as np
from pandas import DataFrame


def check_configure( sample_fdname, subfd_names ):
    # rawdata_folder = f"{sample_fdname}/raw"
    result_folder = f"{sample_fdname}/results"
    if not exists(result_folder):
        makedirs(result_folder)
        print("Create results Directory!")
    else:
        print("Results Directory Exist, Keep going!")

    for subfd in subfd_names:
        subfd = f"{result_folder}/{subfd}"
        if not exists(subfd):
            makedirs(subfd)
            print(f"Create subfolder {subfd} in result!")
        else:
            cover = input("This sample has a record, overwrite it or not (y/n): ")
            if cover.lower() == "y" or cover.lower() == "yes":
                rmtree(subfd)
                makedirs(subfd)
                print(f"Subfolder {subfd} is initialized!")
            else:
                print(f"Results for this sample Exist!")

def create_subfolder( main_fd, sub_fd ):
    fullpath_sub_fd = f"{main_fd}/{sub_fd}"
    if not exists(fullpath_sub_fd):
        makedirs(fullpath_sub_fd)
        print(f"Create subfolder {fullpath_sub_fd} in result!")
    else:
        # cover = input("This sample has a record, overwrite it or not (y/n): ")
        # if cover.lower() == "y" or cover.lower() == "yes":
        #     rmtree(subfd)
        #     makedirs(subfd)
        #     print(f"Subfolder {subfd} is initialized!")
        # else:
        #     print(f"Results for this sample Exist!")
        pass
def check_file_extension( fd_name:str, file_ext:str ):
    """
    return a list of file name with specific file extension
    arg:
        fd_name : searched folder
        file_ext :  Filename Extension
    """
    filename_list = []
    for f in listdir(fd_name):
        fullname = f.split(".")
        extension = fullname[-1]
        name = fullname[0]
        if extension == file_ext and isfile(join(fd_name, f)):
            filename_list.append(f)
    
    return filename_list

def check_subgroup( filename_list, delimiter='_' ):

    sg_list = []
    for fn in filename_list:
        subgroup = fn.split(delimiter)[0]
        sg_list.append(subgroup)

    subgroups, sg_counts = np.unique(sg_list, return_counts=True)
    file_strcture = {}
    # Init dict
    for sg_name in subgroups:
        file_strcture[sg_name] = []

    for fn in filename_list:
        subgroup = fn.split(delimiter)[0]
        subgroup_idx = np.where(subgroups == subgroup)
        #file_idx = int(fn.split("_")[1])
        #if file_idx < sg_counts[subgroup_idx]:
        file_strcture[subgroup].append(fn)
    return file_strcture



def save_power_dep( df:DataFrame, output_name ):
    
    # condi_1 = (df["Qi_dia_corr_err"] / df["Qi_dia_corr"] > 1.0) #| (df["Qi_dia_corr"] < 0) #|(df["Qi_dia_corr_err"] < 1e8)
    # condi_2 = (df["absQc_err"] / df["absQc"] > 1.0) | (df["absQc"] < 0)
    # condi_3 = (df["Ql_err"] / df["Ql"] > 1.0) | (df["Ql"] < 0)
    # indexNames = df[(condi_1 | condi_2 | condi_3)].index 
    # df.drop(indexNames , inplace=True)
    df.to_csv(output_name, index=False)  

def save_tanloss( df:DataFrame, output_name ):

    newColOrder = list(df.columns)
    # newColOrder.remove( "power" )
    # newColOrder.insert(0, "power" )
    # all_results = all_results[newColOrder]
    #dictResult = dfResults.to_dict(orient="list")
    #outfn = fn.replace(".mat","")
        
        
    #df.to_csv(output_name, index=False)
    
    # condi_1 = (df["Qi_dia_corr_err"] / df["Internal Q"] > 0.2) | (df["Internal Q"] < 0) #|(df["Qi_dia_corr_err"] < 1e8)
    # condi_2 = (df["absQc_err"] / df["Coupling Q"] > 0.2) | (df["Coupling Q"] < 0)
    # condi_3 = (df["Ql_err"] / df["Loaded Q"] > 0.2) | (df["Loaded Q"] < 0)
    # indexNames = df[(condi_1 | condi_2 | condi_3)].index 
    # df.drop(indexNames , inplace=True)
    df.to_csv(output_name, index=False)  