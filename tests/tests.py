import numpy

l = numpy.array([
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0,
            15.0]], [[16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]]
        ]).reshape(2, 4, 3)
r = numpy.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]).reshape(4, 3)


# o = numpy.tensordot(l, r, axes=2)
# print(o == numpy.array([506.0, 1298.0]).reshape(2))

def test():
    """Stupid test function"""
    o = numpy.array([0.0, 0.0])
    for i in range(100):
        o += numpy.tensordot(l, r, axes=2)

    if (o != numpy.array([100 * 506.0, 100 * 1298.0])).all():
    	print("ERROR")

if __name__ == '__main__':
    import timeit
    print(timeit.timeit("test()", setup="from __main__ import test", number=10000))