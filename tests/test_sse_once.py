"""Mirrors frontend/src/utils/once.ts — SSE finish must invoke onDone exactly once."""


def once(fn):
    done = False

    def inner():
        nonlocal done
        if done:
            return
        done = True
        fn()

    return inner


def test_finish_calls_on_done_once():
    n = {"c": 0}
    finish = once(lambda: n.__setitem__("c", n["c"] + 1))
    finish()
    finish()
    assert n["c"] == 1
