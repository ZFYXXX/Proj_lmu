import copy

import pytest

from lmms_eval.tasks.ok_vqa.utils import ok_vqa_process_results
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor
from scripts.eval.lmms_eval.run_qwen25_okvqa import answers_match_logged_target


def test_native_scorer_mutates_answers_but_checker_accepts_exact_logged_form():
    answers = ['diner']*6 + ["denny's"]*2 + ['waffle house']*2
    saved = copy.deepcopy(answers)
    doc = {'answers': copy.deepcopy(answers), 'question_id': 1}
    scored = ok_vqa_process_results(doc, ['diner'])
    assert scored['exact_match'] == 1
    assert doc['answers'] != saved
    assert doc['answers'][6] == "denny 's"
    for value in (saved, repr(saved), doc['answers'], repr(doc['answers'])):
        assert answers_match_logged_target(value, saved)
    assert answers == saved


@pytest.mark.parametrize('value', [None, 'malformed', '{"answer": "diner"}', ['diner'],
                                 ['diner', 'diner', 'other'], ['diner', 3, "denny 's"]])
def test_missing_changed_and_malformed_answers_fail(value):
    assert not answers_match_logged_target(value, ['diner', 'diner', "denny's"])


def test_order_and_duplicate_counts_are_preserved():
    raw = ['red', 'blue', 'blue']
    assert not answers_match_logged_target(['blue', 'red', 'blue'], raw)
    assert not answers_match_logged_target(['red', 'blue'], raw)
    assert not answers_match_logged_target(['red', 'blue', 'red'], raw)


def test_only_the_exact_official_normalization_is_accepted():
    raw = ['The two cats', "she's wearing coat"]
    expected = [EvalAIAnswerProcessor()(a) for a in raw]
    assert answers_match_logged_target(expected, raw)
    assert not answers_match_logged_target(['two cats', "she 's wearing coat"], raw)
