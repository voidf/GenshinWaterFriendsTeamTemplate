import math

def solve_quadratic_equation(a, b, c):
    d = b * b - 4 * a * c
    if d < 0:
        return
    d = d**0.5
    m = 1/(2*a)
    return (d-b)*m, (-d-b)*m

class vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
    def __add__(self, rhs):
        return vec3(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    def __sub__(self, rhs):
        return vec3(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    def __mul__(self, rhs):
        return vec3(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    def __truediv__(self, rhs):
        return vec3(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)

    def __and__(self, rhs):
        """点积，dot"""
        return self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    def dot(self, rhs):
        return self & rhs

    def __xor__(self, rhs):
        """叉积，cross"""
        return vec3(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x
        )
    def cross(self, rhs):
        return self ^ rhs

    def mag2(self):
        return self.x * self.x + self.y * self.y + self.z * self.z 
    def mag(self):
        return self.mag2() ** 0.5
    def __repr__(self):
        return f"vec3({self.x}, {self.y}, {self.z})"
    def __str__(self):
        return self.__repr__()

class Sphere:
    def __init__(self, c: vec3, r=0.0):
        self.c = c
        self.r = r
    def __repr__(self):
        return f"Sphere({self.c}, {self.r})"
    def __str__(self):
        return self.__repr__()

if __name__ == '__main__':
    a = vec3(2, 1, 3)
    b = vec3(1, 5, 8)
    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b)
    print(a ^ b)
    print(a & b)