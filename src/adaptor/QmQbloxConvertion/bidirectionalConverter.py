""" Transfrom the raw nc data measured by QM into Qblox form. And it's reversible which means: qm -> qblox -> qm is okay. """
""" Warning: Once you use it, the old nc will be modified and saved with the qblox format. It won't save a new nc for you. """
import os
from abc import ABC, abstractmethod
from xarray import open_dataset, Dataset
from numpy import array, arange


class QQAdapter(ABC):
    """ QM, Qblox dataset bidirectional transformer. """
    def __init__( self ):
        self.__file_path:str = ""
        self.__dataset:Dataset = None
        self.__data_type:str = "nc"
        self.__target_format:str = "qblox" # qblox | qm 
    
    @property
    def file_path( self ):
        return self.__file_path
    @property
    def ds( self ):
        return self.__dataset
    @property
    def data_type( self ):
        return self.__data_type
    @property
    def target_format( self ):
        return self.__target_format

    """ prepare settings, built-in """
    def settings(self, file_path:str, target_format:str=""):
        
        self.__file_path = file_path
        self.__data_type = os.path.split(file_path)[-1].split(".")[-1]

        match self.__data_type:
            case 'nc':
                self.__dataset = open_dataset(self.__file_path)
            case _:
                raise ImportError(f"Data type = {self.__data_type} can't be handled so far.")

        match target_format.lower():
            case "qblox" | "qb" | None | "":
                self.__target_format = "qblox"
            case 'qm':
                self.__target_format = "qm"
            case _:
                raise ValueError(f"Irrecognizable target_format = {target_format} was given.")

       
    """ Executor function, built-in"""     
    def transformExecutor(self):
        
        match self.__target_format:
            case "qm":
                self.__dataset = self.QBtoQM_adapter()
            case "qblox":
                self.__dataset = self.QMtoQB_adapter()

        # 3. overwrite the file with new dataset  
        new_name = os.path.basename(self.__file_path).split(".")[0] + "_QMtoQblox" + f".{self.data_type}"
        self.__dataset.to_netcdf(os.path.join(os.path.split(self.__file_path)[0],new_name))


    """ Develop case by case """
    @abstractmethod
    def QMtoQB_adapter( self ):
        """ transform the dataset. """
        pass

    @abstractmethod
    def QBtoQM_adapter( self ):
        """ transform the dataset. """
        pass


####################################
##########    Examples    ##########
####################################

class FluxCavConverter(QQAdapter):
    """ Use `self.transformExecutor()` and get the dataset by `self.ds` """
    def __init__(self, file_path:str, target_format:str):
        super().__init__()
        self.settings(file_path, target_format)

    def QBtoQM_adapter(self):
        pass

    def QMtoQB_adapter(self)->Dataset:
        dict_ = {}
        bias = array(self.ds.coords["flux"])
        for q_ro in self.ds.data_vars:
            q = q_ro.split("_")[0]
            freq_values = 2*bias.shape[0]*list(array(self.ds.coords["frequency"]))
            dict_[q] = (["mixer","bias","freq"],array(self.ds[q_ro]).reshape(2,bias.shape[0],array(self.ds.coords["frequency"]).shape[0]))
            dict_[f"{q}_freq"] = (["mixer","bias","freq"],array(freq_values).reshape(2,bias.shape[0],array(self.ds.coords["frequency"]).shape[0]))
         
        return Dataset(dict_,coords={"mixer":array(["I","Q"]),"bias":bias,"freq":arange(array(self.ds.coords["frequency"]).shape[0])},attrs={"execution_time":"H00M00S00"})

if __name__ == "__main__":
    FCT = FluxCavConverter('/Users/ratiswu/Downloads/qm experiment data/Find_Flux_Period.nc','qblox')
    FCT.transformExecutor()
    print(FCT.ds)