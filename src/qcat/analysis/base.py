from abc import ABC, abstractmethod
from datetime import datetime
from xarray import Dataset

class QCATAna( ABC ):

    def __init__( self, silence_mode:bool=False ):
        self.silence_mode = silence_mode
        if self.silence_mode: self.__describe()
        self.raw_data = None
        self.result = None


    @abstractmethod
    def _import_data( self, *args, **kwargs ):
        """ Used to check input data """
        pass


    @abstractmethod
    def _start_analysis( self, *args, **kwargs ):
        """ Used to start analysis which might be time costly """
        pass

    @abstractmethod
    def _export_result( self, *args, **kwargs ):
        """ Export result with a format from analysis"""
        pass

    # def run( self, save_path:str = None ):
        
    #     self.raw_data = self._import_data()

    #     if self.raw_data is not None:
    #         self.result = self._start_analysis()

    #         if save_path is not None:
    #             self.save_path = save_path
    #             self._export_result()

    #         return self.result

    #     else:
    #         print("Import data failed.")

    

    

    def __describe(self):
        return f"Initializing {self.__class__.__name__}"
    
    # def close(self):
    #     print("Analysis is finished.")

    # def __del__(self):
    #     print("QCAT object has been deleted")
    #     try:
    #         self.close()

    #     except AttributeError:
    #         # In case __inst was not initialized correctly
    #         print("self.close() failed.")
    #         pass