from triton.experimental.tle.language.gpu.types import distributed_encoding


class RegisterOrder(distributed_encoding):
    def __init__(self, reg, lane, warp, shape):
        self.reg = tuple(map(tuple, reg))
        self.lane = tuple(map(tuple, lane))
        self.warp = tuple(map(tuple, warp))
        self.shape = tuple(shape)

    @property
    def rank(self):
        return len(self.shape)

    def to_ir(self, builder):
        return builder.get_tle_linear_encoding(self.reg, self.lane, self.warp, [], self.shape)

    def __hash__(self):
        return hash((self.reg, self.lane, self.warp, self.shape))

    def __eq__(self, x):
        return type(x) == type(self) and (self.reg, self.lane, self.warp, self.shape) == (
            x.reg,
            x.lane,
            x.warp,
            x.shape,
        )

    def __repr__(self):
        return "RegisterOrder" + repr((self.reg, self.lane, self.warp, self.shape))
