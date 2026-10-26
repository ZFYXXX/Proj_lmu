"""Label-free matched wrong-question assignment for selector-only intervention."""
import hashlib

FORMAT = 'qwen25_selector_condition_manifest_v1'
GENERIC = {
    'chartqa': 'Read the chart.\nAnswer the question with a single word.',
    'ocrbench': 'Read the text in the image.',
    'ai2d': "Answer the diagram question.\nAnswer with the option's letter from the given choices directly.",
}


def request_key(sample_id):
    # The fourth field is evaluator processing order, NOT a per-document zero.
    # Smoke and full runs have different processing-order indices.
    parts = str(sample_id).split(':')
    if len(parts) != 4 or not parts[2].isdigit() or not parts[3].isdigit():
        raise ValueError('Expected native task:split:doc_id:request_index identity')
    return ':'.join(parts[:3])


def ids_sha(ids):
    import numpy as np
    return hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()


def text_ids(tokenizer, text):
    values = list(tokenizer.encode(text, add_special_tokens=False))
    # Match native last-user extraction: remove newline tokens adjacent to
    # visual sentinels, not arbitrary text whitespace/tokenizer normalization.
    while values and values[0] == 198:
        values.pop(0)
    while values and values[-1] == 198:
        values.pop()
    if not values:
        raise ValueError('Empty selector condition')
    return values


def matched_donors(rows, *, seed=42):
    """Within-stratum permutation, other image, maximize changed questions.

    Lexicographic costs: avoid same-image donors; maximize different question
    IDs; minimize total length mismatch; seeded tie-break. This is matched
    substitution, not an unconditional uniformly random permutation.
    Identical-question strata remain explicitly non-informative.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row['task'], row['stratum'])].append(index)
    output = [None] * len(rows)
    rng = np.random.default_rng(seed)
    for _, indices in sorted(groups.items()):
        n = len(indices)
        images = [rows[i]['image_fingerprint'] for i in indices]
        if n < 2 or len(set(images)) < 2:
            for i in indices:
                output[i] = {'donor_index': i, 'informative': False, 'reason': 'no_other_image_in_stratum'}
            continue
        lengths = np.array([len(rows[i]['true_ids']) for i in indices])
        questions = np.array([ids_sha(rows[i]['true_ids']) for i in indices])
        same_image = np.asarray(images)[:, None] == np.asarray(images)[None, :]
        same_question = questions[:, None] == questions[None, :]
        question_cost = n * (int(lengths.max()) + 1) + 1
        image_cost = question_cost * (n + 1)
        costs = (same_image * image_cost + same_question * question_cost +
            np.abs(lengths[:, None] - lengths[None, :])).astype(np.float64)
        costs += rng.random((n, n)) / (n + 1)
        sources, donors = linear_sum_assignment(costs)
        for source, donor in zip(sources, donors):
            i, j = indices[int(source)], indices[int(donor)]
            if same_image[source, donor]:
                raise ValueError('Cannot construct an other-image permutation in this stratum')
            informative = not bool(same_question[source, donor])
            output[i] = {'donor_index': j, 'informative': informative,
                'reason': 'different_question' if informative else 'identical_question_after_matching'}
    if any(row is None for row in output):
        raise ValueError('Incomplete donor assignment')
    return output
