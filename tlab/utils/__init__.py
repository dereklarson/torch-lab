class NameRepr(type):
    def __repr__(cls):
        return cls.__name__


def param_repr(param):
    try:
        return param.__name__
    except AttributeError:
        return param
