import math


class Vector3:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z
        pass

    def __str__(self):
        return f"({self.x},{self.y},{self.z})"

    def __add__(self, other):
        if type(other) is Vector3:
            return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
        elif type(other) is float or int:
            return Vector3(self.x + other, self.y + other, self.z + other)
        else:
            return self

    def __radd__(self, other):
        if type(other) is float or int:
            return Vector3(self.x + other, self.y + other, self.z + other)
        else:
            return self

    def __sub__(self, other):
        if type(other) is Vector3:
            return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return self

    def __mul__(self, other):
        if type(other) is Vector3:
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        elif type(other) is float or int:
            return Vector3(self.x * other, self.y * other, self.z * other)
        else:
            return self

    def __rmul__(self, other):
        if type(other) is float or int:
            return Vector3(self.x * other, self.y * other, self.z * other)
        else:
            return self

    def __truediv__(self, other):
        if type(other) is Vector3:
            return Vector3(self.x / other.x, self.y / other.y, self.z / other.z)
        elif type(other) is float or int:
            return Vector3(self.x / other, self.y / other, self.z / other)
        else:
            return self

    def length(self) -> float:
        return abs(math.sqrt((self.x**2) + (self.y**2) + (self.z**2)))
