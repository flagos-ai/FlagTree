from __future__ import annotations


class InOut(object):

    @classmethod
    def __class_getitem__(cls, desc: str) -> str:
        return desc


class Input(object):

    @classmethod
    def __class_getitem__(cls, desc: str) -> str:
        return desc
