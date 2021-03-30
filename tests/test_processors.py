from adfluo.processors import param, SampleProcessor


def test_proc_params():
    class TestProc(SampleProcessor):
        def __init__(self, a, b):
            self.a = param(a)
            self.b = param(b)

    assert TestProc(1, 2) == TestProc(1, 2)
    assert TestProc(1, "a") == TestProc(1, "b")
