import math
import random as rng
import matplotlib.pyplot as plt


class Vector2:

    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_magnitude(self):
        return math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2))

    def add_vector(self, vector):
        return Vector2(self.x + vector.x, self.y + vector.y)

    def subtract_vector(self, vector):
        return Vector2(self.x - vector.x, self.y - vector.y)

    def distance(self, vector):
        return self.subtract_vector(vector).get_magnitude()

    def divide(self, n):
        if n != 0:
            return Vector2(self.x/n, self.y/n)

    def get_vector(self):
        return (self.x, self.y)

    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    @staticmethod
    def get_random(x_min=-200, x_max=200, y_min=-200, y_max=200):
        return Vector2(rng.randint(x_min, x_max), rng.randint(y_min, y_max))


class Instance:

    attributes = Vector2(0, 0)

    def __init__(self, attributes):
        # TODO check valid
        self.attributes = attributes

    def add_instance(self, other):
        return Instance(self.attributes.add_vector(other.attributes))

    def distance(self, other):
        # obtain distance to other instance
        return self.attributes.distance(other.attributes)

    def divide(self, n):
        return Instance(self.attributes.divide(n))

    def get_vector(self):
        return self.attributes.get_vector()

    def __str__(self):
        return str(self.attributes)

    @staticmethod
    def get_random():
        return Instance(Vector2.get_random())


class KMeans:

    k = 0
    instances = None # instance array (x)
    centroids = None # centroid array (m)
    prev_centroids = None
    belonging_bits = None # bool matrix (rows: instances, cols: centroids) (matrix[row][col])
    tolerance = 0
    max_it = 0

    def __init__(self, k, instances, tolerance = 0.1, max_it = 100):
        # TODO check valid
        self.k = k
        self.instances = instances
        self.centroids = [None] * k
        self.prev_centroids = self.centroids.copy()
        self.belonging_bits = [[False for i in range(k)] for t in range(len(instances))]
        self.tolerance = tolerance
        self.max_it = max_it

    def do_the_thing(self):
        # initialize centroids to random instances
        self._init_centroids()
        self._check_print(bits=False, instances=True)
        it = 0
        # iterate until centroids converge
        while it < self.max_it and not self._check_finished():
            it += 1
            # update belonging matrix
            self._set_bits()
            # self._plot() uncomment to plot every iteration
            # obtaint new centroids
            self._update_centroids()

        # obtain final belonging matrix
        self._set_bits()
        # print retults
        print("iterations: {}".format(it))
        if it == self.max_it:
            print("max iteration number reached!")
        self._check_print(prev=True)
        self._plot()

    def _init_centroids(self):
        for i in range(0, self.k):
            self.centroids[i] = self._get_random_instance(self.centroids)

    def _update_centroids(self):
        # save previous centroids for furture comparison
        self.prev_centroids = self.centroids.copy()

        for i in range(0, len(self.centroids)):
            bits_i = 0 # nº instances that belong to centroid i
            instance_sum = None # sum of those instances
            for t in range(0, len(self.instances)):
                if self.belonging_bits[t][i]:
                    bits_i += 1
                    if instance_sum == None:
                        instance_sum = self.instances[t]
                    else:
                        instance_sum = instance_sum.add_instance(self.instances[t])
            self.centroids[i] = instance_sum.divide(bits_i)

        # reset belonging matrix
        self.belonging_bits = [[False for i in range(len(self.centroids))] for t in range(len(self.instances))]

    def _get_random_instance(self, used):
        instance = used[0]
        while instance in used:
            instance = self.instances[rng.randint(0, len(self.instances) - 1)]
        return instance

    def _set_bits(self):
        for t in range(0, len(self.instances)):
            min = 99999
            min_i = -1
            for i in range(0, len(self.centroids)):
                dif = self.instances[t].distance(self.centroids[i])
                if dif < min:
                    min = dif
                    min_i = i
            self.belonging_bits[t][min_i] = True

    def _check_finished(self):
        finished = True
        for i in range(0, (len(self.centroids))):
            centroid = self.centroids[i]
            prev = self.prev_centroids[i]
            finished = prev != None and centroid.distance(prev) < self.tolerance
            if not finished:
                break
        return finished

    def _plot(self):
        plotter = KMeansPlotter(self.instances, self.centroids, self.belonging_bits)
        plotter.plot()

    def _check_print(self, k=True, instances=False, centroids=True, bits=True, prev=False):
        if k:
            print("k: {}\n".format(self.k))
        if instances:
            print("instances:")
            t = 0
            for instance in self.instances:
                print("\tt: {} -> {}".format(t, instance))
                t += 1
            print()
        if centroids:
            print("centroids:")
            i = 0
            for centroid in self.centroids:
                print("\ti: {} -> {}".format(i, centroid))
                i += 1
            print()
        if prev:
            print("prev_centroids:")
            i = 0
            for centroid in self.prev_centroids:
                print("\ti: {} -> {}".format(i, centroid))
                i += 1
            print()
        if bits:
            print("bits:")
            for t in range(len(self.instances)):
                print("\tt: {} -> {}: ".format(t, self.instances[t]))
                for i in range(len(self.centroids)):
                    print("\t\ti: {} -> {}: {}".format(i, self.centroids[i], self.belonging_bits[t][i]))

    @staticmethod
    def test():
        instance_num = rng.randint(100, 600)
        # instance_num = 100
        k = rng.randint(2, 7)
        # k = 7
        instances = [Instance.get_random() for i in range(0, instance_num)]
        k_means = KMeans(k, instances)
        k_means.do_the_thing()


class KMeansPlotter:

        instances = None # instance array (x)
        centroids = None # centroid array (m)
        belonging_bits = None # bool matrix (rows: instances, cols: centroids) (matrix[row][col])

        def __init__(self, instances, centroids, belonging_bits):
            self.instances = instances
            self.centroids = centroids
            self.belonging_bits = belonging_bits

        def _get_color_letter(index):
            if index == 0:
                return 'b'
            elif index == 1:
                return 'g'
            elif index == 2:
                return 'r'
            elif index == 3:
                return 'c'
            elif index == 4:
                return 'm'
            elif index == 5:
                return 'y'
            elif index == 6:
                return 'k'
            else:
                return ''

        def plot(self):
            index = 0
            for instance in self.instances:
                vector = instance.get_vector()
                bit_row = self.belonging_bits[index]
                col = 0
                for bit in bit_row:
                    if bit:
                        break
                    col += 1
                plt.plot(vector[0], vector[1], "{}o".format(KMeansPlotter._get_color_letter(col)))
                index += 1

            index = 0
            for centroid in self.centroids:
                vector = centroid.get_vector()
                plt.plot(vector[0], vector[1], "{}+".format(KMeansPlotter._get_color_letter(index)))
                index += 1

            plt.show()


if __name__ == "__main__":
    KMeans.test()
