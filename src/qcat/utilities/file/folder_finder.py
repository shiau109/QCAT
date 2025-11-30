import os

def get_child_folders(parent_path, pattern=None):
    """
    Get list of child folder names that contain the specified pattern.
    
    Parameters:
    -----------
    parent_path : str
        Path to the parent directory to search in
    pattern : str or None, default None
        Pattern to match in folder names. If None, returns all folders.
        
    Returns:
    --------
    list
        List of folder names (not full paths) that contain the pattern,
        or all folders if pattern is None
    """
    if not os.path.exists(parent_path):
        print(f"Warning: Path does not exist: {parent_path}")
        return []
    
    child_folders = []
    try:
        # Get all items in the parent directory
        for item in os.listdir(parent_path):
            item_path = os.path.join(parent_path, item)
            # Check if it's a directory
            if os.path.isdir(item_path):
                # If no pattern specified, include all folders
                # If pattern specified, only include folders containing the pattern
                if pattern is None or pattern in item:
                    child_folders.append(item)
        
        # Sort the folders for consistent ordering
        child_folders.sort()
        
    except PermissionError:
        print(f"Permission denied accessing: {parent_path}")
    except Exception as e:
        print(f"Error reading directory {parent_path}: {e}")
    
    return child_folders

def get_child_folder_paths(parent_path, pattern=None):
    """
    Get list of full paths to child folders that contain the specified pattern.
    
    Parameters:
    -----------
    parent_path : str
        Path to the parent directory to search in
    pattern : str or None, default None
        Pattern to match in folder names. If None, returns all folders.
        
    Returns:
    --------
    list
        List of full paths to folders that contain the pattern,
        or all folders if pattern is None
    """
    folder_names = get_child_folders(parent_path, pattern)
    return [os.path.join(parent_path, name) for name in folder_names]