class Config:
    def __init__(self, **kwargs):
        self._set_attributes(**kwargs)

    def _set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_attributes(self, **kwargs):
        self._set_attributes(kwargs)