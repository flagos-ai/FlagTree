class UnknownSymbolError(Exception):

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(f"unknown symbol {name}", *args, **kwargs)
