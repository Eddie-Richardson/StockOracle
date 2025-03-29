# app/utils/normalization.py
def normalize_param(param, default=None):
    """
    Returns the parameter stripped of whitespace.
    If the parameter is None or an empty string after stripping, returns the default value.
    """
    if param is None or str(param).strip() == "":
        return default
    return str(param).strip()
