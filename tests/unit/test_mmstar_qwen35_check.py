import pytest
from configs.eval.lmms_eval.tasks.mmstar_qwen35_check.mmstar_check_utils import extract, prompt, results


@pytest.mark.parametrize('output,expected', [('A','A'), ('b.','B'), ('(C)','C'), ('{"answer":"d"}','D'),
    ('The answer is B','B_NONE'), ('Because B is longer','B_NONE'), ('ABCD','B_NONE'), ('(A','B_NONE'), ('A)','B_NONE'),
    ('{"answer":"A","reason":"B"}','B_NONE'), ('','B_NONE')])
def test_only_unambiguous_whole_response_is_scored(output, expected):
    assert extract(output) == (None if expected == 'B_NONE' else expected)


def test_prompt_never_reads_answer_and_invalid_does_not_guess():
    assert prompt({'question':'Question?', 'answer':'A'}) == prompt({'question':'Question?', 'answer':'B'})
    assert results({'answer':'B'}, ['Because B is longer']) == {'strict_accuracy':0., 'valid_answer':0.}
