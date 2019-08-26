def consecutive_couples(iterable):
    firsts = list(iterable)
    firsts.pop(-1)
    seconds = list(iterable)
    seconds.pop(0)
    for first, second in zip(firsts, seconds):
        yield first, second
