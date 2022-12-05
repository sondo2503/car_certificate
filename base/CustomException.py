class Error(Exception):
    """Base class for other exceptions"""
    pass

class EmptyResultException(Error):
    """Raised when the input value is too small"""
    pass

class BlurrImgException(ValueError):
    pass