# from https://nessy.info/?p=16
# modified to fit sdt axis mapping

class Line(object):

    def __init__(self, data):
        self.first, self.second = data
        self.s = self.slope()
        self.y_inter = self.yintercept(self.s)
        print(self.s, self.y_inter)

    def slope(self):
        '''Get the slope of a line segment'''
        (x1, y1), (x2, y2) = self.first, self.second
        try:
            return (float(y2)-y1)/(float(x2)-x1)
        except ZeroDivisionError:
            # line is vertical
            return None

    def yintercept(self, slope):
        '''Get the y intercept of a line segment'''
        if slope != None:
            x, y = self.first
            return y - slope * x
        else:
            return None

    def solve_for_y(self, x, slope, yintercept):
        '''Solve for Y cord using line equation'''
        if slope != None and yintercept != None:
            return float(slope) * x + float(yintercept)
        else:
            raise Exception('Can not solve on a vertical line')

    def solve_for_x(self, y, slope, yintercept):
        '''Solve for X cord using line equation'''
        if slope != 0 and slope:
            return float((y - float(yintercept))) / float(slope)
        else:
            raise Exception('Can not solve on a horizontal line')

    def solve(self, x):
        '''find y given x using default slopw and y-intercept'''
        if self.s != None and self.y_inter != None:
            return float(self.s) * x + float(self.y_inter)
        else:
            raise Exception('cannot solve')