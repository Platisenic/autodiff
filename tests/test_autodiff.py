import autodiff
from pytest import approx

class TestAutoDiff:
    def test_base_1(self):
        a = autodiff.vec([1, 2, 3, 4, 5])
        a[0] = 10
        a[1] = 15

        assert len(a) == 5
        assert isinstance(a, autodiff.vec)
        assert a.values() == [10, 15, 3, 4, 5]
        assert a[1] == 15
    
    def test_base_2(self):
        a = autodiff.vec(3)
        for i in range(len(a)):
            a[i] = i+20
        
        assert len(a) == 3
        assert isinstance(a, autodiff.vec)
        assert a.values() == [20, 21, 22]
        assert a[1] == 21

    def test_grad_1(self):
        a = autodiff.vec([1, 2, 3, 4, 5])
        b = autodiff.vec([5, 4, 3, 2, 1])
        Q = 3 * a * a - b * b
        Q.backward()

        for grad, gold in zip(a.grad(), (6 * a).values()):
            assert grad - gold == approx(0)
        
        for grad, gold in zip(b.grad(), (-2 * b).values()):
            assert grad - gold == approx(0)


    def test_grad_2(self):
        a = [i for i in range(10)]
        b = [i*2+1 for i in range(10)]
        a = autodiff.vec(a)
        b = autodiff.vec(b)
        Q = a.sin() + b.cos() + a * 5 - (b+2)
        Q.backward()

        for grad, gold in zip(a.grad(), (a.cos() + 5).values()):
            assert grad - gold == approx(0)
        
        for grad, gold in zip(b.grad(), ((-1) * b.sin() - 1).values()):
            assert grad - gold == approx(0)

    
    def test_grad_3(self):
        a = [i+1 for i in range(10)]
        b = [i*2+1 for i in range(10)]
        a = autodiff.vec(a)
        b = autodiff.vec(b)
        Q = a.log() * b.exp()
        Q.backward()

        for grad, gold in zip(a.grad(), (1/a * b.exp()).values()):
            assert grad - gold == approx(0)
        
        for grad, gold in zip(b.grad(), (a.log() * b.exp()).values()):
            assert grad - gold == approx(0)

