__author__ = 'anton'


class Transformation(object):
    def __init__(self, domain_x, domain_y, range_x, range_y, scale, offset,
                 range_size, domain_size, transform_type, is_flat=False):
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.range_x = range_x
        self.range_y = range_y
        self.scale = scale
        self.offset = offset
        self.range_size = range_size
        self.domain_size = domain_size
        self.transform_type = transform_type
        self.is_flat = is_flat

    @staticmethod
    def from_struct(range_size, domain_size, struct):
        return Transformation(struct[0], struct[1], struct[2], struct[3], struct[4],
                              struct[5], range_size, domain_size, struct[6])
