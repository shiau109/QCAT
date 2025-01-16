""" Protocols for raw data dataset. Coordinates, data variables, etc."""
from xarray import Dataset
from numpy import array, ndarray

""" Register the data shape in `DSCoordNameRegister` first """
class DSCoordNameRegister():
    def __init__( self ):
        pass
    
    # register zone start
    #TODO: Register the Exp_name starts with a '_', and upper or lower font are both okay.
    def _FluxCavity( self ):
        return ["mixer", "bias", "freq"]

    # register zone end
    
    # do NOT touch !
    def get_coordnameANDshape(self, exp_name:str)->list:
        registered_exp_name = [(attr.lower(), getattr(self, attr)) for attr in dir(self) if  attr.startswith("_") and not attr.endswith("_") and callable(getattr(self, attr))]
        
        ans = []
        for exp in registered_exp_name:
            if exp[0].replace("_","") == exp_name.lower():
                ans = exp[1]()
        if len(ans) == 0:
            raise NameError(f"The exp name '{exp_name}' haven't been registered. ")

        print(f"** The shape for EXP '{exp_name}' = {ans}.\n    => Please assign the coords in dataset base on them.")

        return ans  

    

""" Use it to build your dataset"""
class DatasetCompiler():
    def __init__( self, exp_name:str, RegisterReference:bool=True):
        """ 
        ### arg explains:\n
        1. `exp_name` must be given, please refer to `DSCoordNameRegister()` for the name you can assign. If you don't wanna use a registered exp name, set it '' or None.\n
        2. `RegisterReference`, go `DSCoordNameRegister()` to get the shape base on what the `exp_name` you give. If you don't wanna use a registered exp name, set it False.\n
        \n
        ## * Warning:\n 
            If you set `RegisterReference = False`, You must needa use `self.change_varShape()` to assign a new data shape.
        """

        self.__checkpoint__:bool = True
        self.__coordinates:dict = {}
        if RegisterReference:
            self.__shape:list = DSCoordNameRegister().get_coordnameANDshape(exp_name) # It will print out the Dataset.data_var shape for the dataset, and it also shows you the name for coords.
        else:
            self.__shape:list = []
        
        self.attributes:dict = {}
        self.data_vars:dict = {}
        self.__output_dataset:Dataset = None

    @property
    def coords( self ):
        return self.__coordinates
    @property
    def var_shape( self ):
        return self.__shape
    @property
    def dataset( self ):
        return self.__output_dataset
    
    def __check_coords_match_shape__( self ):
        """ Check the name in coordinates match the shape """

        if self.__checkpoint__:
            
            if self.__coordinates == {}:
                raise ValueError("The coordinate is empty !")
            if self.__shape == []:
                raise ValueError("The data shape is empty !")
            
            if set(self.__shape) != set(self.__coordinates.keys()):
                print(f"Coords: {self.__coordinates}")
                print(f"Shape : {self.__shape}")
                raise SyntaxError(f"The given coordinates and data shape is mismatched !")
            
            self.__checkpoint__ = False
    
    def __check_data_shape_matched__(self, data:ndarray, name:str=""):
        """ Check the given data array shape is match or not. """
        
        if self.__shape == []:
            raise ValueError("The data shape is empty !")

        right_shape = []
        for coord_name in self.__shape:
            right_shape.append(array(self.__coordinates[coord_name]).shape[0])

        if data.shape != tuple(right_shape):
            print("The data shape: ",data.shape)
            print("The target shape: ",tuple(right_shape))
            raise IndexError(f"The shape of your given data '{name}' is not what you assign before, Check it !")
            

    def assign_coords(self, coordinates:dict):
        """ Fill in your coordinates in dict here. """
        
        if isinstance(coordinates,dict):
            if len(list(coordinates.keys())) != 0:
                self.__checkpoint__ = True
                self.__coordinates = coordinates
            else:
                raise ValueError("We reject the empty coordinates !")
        else:
            raise TypeError(f"Coordinates must be a dict, but {type(coordinates)} was the given type.")
    
    def change_varShape(self, shape:list):
        """ Basically, the shape will be given by `DSCoordNameRegister()` with the exp name you initialized. Use this if you want to change it. """

        if isinstance(shape,list):
            if len(shape) != 0:
                self.__checkpoint__ = True
                self.__shape = shape
            else:
                raise ValueError("We reject the empty data shape !")
        else:
            raise TypeError(f"Data shape must be a tuple or list, but {type(shape)} was the given type.")
        
        
    def add_data(self, data_var:dict):
        """ data_var is a dict, it's key will be the data_var name and value will be the data for the xarray.Dataset. 
            #### Warning: Make sure all the data are in the same shape as what you give by `self.set_dataShape()`.
        """
        if not isinstance(data_var,dict):
            raise TypeError("While you are adding the data vars, arg: `data_var` must be a dict.")
        
        
        self.__check_coords_match_shape__()

        for name in data_var:
            value = array(data_var[name])
            self.__check_data_shape_matched__(value, name)
            self.data_vars[name] = (self.__shape, value)

    def add_attrs(self, new_attributes:dict, old_attributes:dict={}):
        """ Assign the  attributes in a dict here. If `Old_attributes` was given, we merge them together and put `old_attributes` at the first priority when there is a key name conflict. """
        if not isinstance(new_attributes,dict):
            raise TypeError("While you are adding the new attributes, arg: `new_attributes` must be a dict.")

        if not isinstance(old_attributes,dict):
            raise TypeError("While you are adding the old attributes, arg: `old_attributes` must be a dict.")

        if old_attributes != {}:
            new_attributes.update(old_attributes) # old attr is at the first priority when there is a keyname conflict.

        self.attributes.update(new_attributes)

    def export_dataset(self,complete_save_path:str=None)->Dataset:
        """ Get your dataset. It will be saved to the given `complete_save_path` if it's not None. """
        # final checks
        self.__check_coords_match_shape__()
        for data_name in self.data_vars:
            self.__check_data_shape_matched__(array(self.data_vars[data_name][-1]), data_name)

        # build dataset
        self.__output_dataset = Dataset(data_vars=self.data_vars, coords=self.__coordinates, attrs=self.attributes)
        
        # optional save
        if complete_save_path is not None:
            self.__output_dataset.to_netcdf(complete_save_path)

        return self.__output_dataset



if __name__ == "__main__":
    DC = DatasetCompiler("fluxcavity")    
    
            
