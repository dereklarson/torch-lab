class NameRepr(type):
    def __repr__(cls):
        return cls.__name__
