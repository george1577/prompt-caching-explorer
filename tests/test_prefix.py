import pytest

from cache_lab.prefix import change_one_char, shared_prefix_len


def test_shared_prefix_len():
    assert shared_prefix_len([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert shared_prefix_len([1, 2], [1, 2, 3]) == 2
    assert shared_prefix_len([5], [1]) == 0


def test_change_one_char():
    new, pos = change_one_char("abc def", 0.5)
    assert len(new) == 7 and new != "abc def"
    assert sum(a != b for a, b in zip(new, "abc def")) == 1
    assert new[pos] in "XY"


@pytest.mark.slow
def test_kv_reuse_gives_identical_output():
    from cache_lab.lm import LanguageModel
    from cache_lab.prefix import run_prefix_demo

    lm = LanguageModel()
    system = "You are a careful assistant. " * 20
    run = run_prefix_demo(
        lm,
        [{"role": "system", "content": system}, {"role": "user", "content": "Name a colour."}],
        [{"role": "system", "content": system}, {"role": "user", "content": "Name a fruit."}],
        max_new_tokens=10, repeats=1,
    )
    assert run.shared > 100
    assert run.recomputed < 20
    assert run.outputs_match
    assert run.max_logit_diff < 1e-3
