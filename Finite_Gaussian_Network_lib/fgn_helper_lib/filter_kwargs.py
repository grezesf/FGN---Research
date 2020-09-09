import inspect

def filter_kwargs(dict_to_filter, thing_with_kwargs):
    
    # given a func, filters a dict (typically kwargs) to be passed to func
    # used to passed kwargs to a func that doesn't accept kwargs
    # from https://stackoverflow.com/questions/26515595/how-does-one-ignore-unexpected-keyword-arguments-passed-to-a-function
    sig = inspect.signature(thing_with_kwargs)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_dict = {filter_key:dict_to_filter[filter_key] for filter_key in filter_keys}
    
    return filtered_dict