"""Conservative de-duplication rules (factcheck_video.textmatch): what may and may never be merged."""

from __future__ import annotations

import pytest

from factcheck.validation import default_is_duplicate
from factcheck_video.textmatch import claims_duplicate, critical_signature, loose_key, statements_equivalent

# ------------------------------------------------------------------ must merge


@pytest.mark.parametrize("a,b", [
    ("Water boils at 100 degrees Celsius at sea level.", "WATER BOILS AT 100 DEGREES CELSIUS AT SEA LEVEL"),
    ("The Eiffel Tower was completed in 1889.", "the eiffel tower was completed in 1889"),
    ("The Eiffel Tower was completed in 1889.", "The Eiffel Tower was completed in 1889!"),
    ("The tower is 330 metres tall", "The tower is 330 metres tall."),
    ("Unemployment fell to 3.5% in March 2024.", "Unemployment fell to 3.5 % in March 2024"),
    ("Mount Everest is 8,849 metres tall.", "Mount Everest is 8849 metres tall"),
    ("The Pacific is the largest ocean on Earth.", "Pacific is the largest ocean on the Earth."),
])
def test_equivalent_statements_merge(a, b):
    assert statements_equivalent(a, b)
    assert claims_duplicate((a, a), (b, b))


# ------------------------------------------------------------------ must never merge


@pytest.mark.parametrize("a,b,why", [
    ("The Great Wall of China is visible from the Moon.", "The Great Wall of China is not visible from the Moon.",
     "negation"),
    ("The vaccine is safe for children.", "The vaccine isn't safe for children.", "contraction negation"),
    ("The Eiffel Tower was completed in 1889.", "The Eiffel Tower was completed in 1887.", "year"),
    ("The bridge is 3.2 km long.", "The bridge is 3.2 miles long.", "unit"),
    ("Water boils at 100 degrees Celsius.", "Water boils at 100 degrees Fahrenheit.", "unit (temperature)"),
    ("Sales rose 5 percent in 2023.", "Sales rose 5 percent in 2022.", "date"),
    ("The election was held on 4 March.", "The election was held on 4 April.", "month"),
    ("Paris is the capital of France.", "Lyon is the capital of France.", "named entity"),
    ("The summit was held in Geneva in 2019.", "The summit was held in Vienna in 2019.", "place"),
    ("The new policy was approved by Congress last year.", "The new policy was approved by Parliament last year.",
     "institution"),
    ("Prices rose after the announcement.", "Prices rose before the announcement.", "temporal relation"),
    ("More people voted in 2020 than in 2016.", "Fewer people voted in 2020 than in 2016.", "comparative"),
    ("The airline ordered the A380 aircraft.", "The airline ordered the A350 aircraft.", "model code"),
    ("Vitamin C prevents colds.", "Vitamin D prevents colds.", "single-letter name"),
    ("The company employs 1,200 people.", "The company employs 12,000 people.", "magnitude"),
    ("Only 10 countries signed the treaty.", "10 countries signed the treaty.", "quantifier"),
    ("Biden signed the infrastructure bill into law at a ceremony in the White House Rose Garden with governors",
     "Trump signed the infrastructure bill into law at a ceremony in the White House Rose Garden with governors",
     "sentence-initial name in a long, otherwise identical sentence"),
    ("The government raised taxes on imported cars from Japan and Korea in the budget announced last spring",
     "The government lowered taxes on imported cars from Japan and Korea in the budget announced last spring",
     "antonym verb in a long sentence"),
    ("Dogs chase cats.", "Cats chase dogs.", "word order"),
    ("The drug may cause liver damage.", "The drug will cause liver damage.", "modality"),
    ("Pluto is the ninth planet.", "Pluto was the ninth planet.", "tense"),
    ("Tea or coffee raises blood pressure.", "Tea and coffee raise blood pressure.", "logic word"),
])
def test_meaning_changing_differences_never_merge(a, b, why):
    assert not statements_equivalent(a, b), why
    assert not claims_duplicate((a, a), (b, b)), why


def test_shared_vocabulary_alone_does_not_merge():
    a = "The Pacific Ocean is the largest and deepest ocean on Earth."
    b = "The Atlantic Ocean is the second largest ocean on Earth."
    assert not statements_equivalent(a, b)


@pytest.mark.parametrize("a,b", [
    ("The Eiffel Tower was finished in 1889.", "The Eiffel Tower was completed in 1889."),  # synonym
    ("Water freezes at 0 degrees Celsius.", "At 0 degrees Celsius, water freezes."),  # reordering
    ("The Amazon is the largest river by volume.", "The Amazon is the largest river by discharge volume."),
])
def test_paraphrases_are_kept_apart_by_design(a, b):
    # Safe misses: both claims are checked (extra cost), nothing is lost.
    assert not statements_equivalent(a, b)


def test_same_statement_but_different_source_excerpts_is_kept():
    # The model wrote one statement for two sources whose figures differ: the excerpt check keeps both.
    stmt = "The Eiffel Tower was completed in 1889."
    assert not claims_duplicate((stmt, "completed in 1889"), (stmt, "COMPLETED IN 1887"))
    assert claims_duplicate((stmt, "The Eiffel Tower was completed in 1889."), (stmt, "EIFFEL TOWER COMPLETED IN 1889"))


def test_excerpt_negation_difference_is_kept():
    stmt = "The Great Wall of China is visible from the Moon."
    assert not claims_duplicate((stmt, "It is a myth that the Great Wall is not visible"),
                                (stmt, "The Great Wall is visible from the Moon"))


def test_loose_key_and_signature():
    assert loose_key("  Water   boils, at 100°C! ") == loose_key("WATER BOILS AT 100 C")
    assert loose_key("1889") != loose_key("1887")
    assert critical_signature("It is not 5 km") != critical_signature("It is 5 km")
    assert critical_signature("5 km") != critical_signature("5 miles")
    assert critical_signature("1,000 people") == critical_signature("1000 people")
    # "may" the verb is not treated as a month
    assert critical_signature("It may rain") == critical_signature("It may rain today")


def test_text_workflow_duplicate_rule_is_unchanged():
    # Regression: the default rule (used by `factcheck` and the image workflow) behaves exactly as before:
    # Jaccard >= 0.8 on content tokens and identical figures; it does not look at negation or excerpts.
    s = "The Eiffel Tower was completed in 1889."
    assert default_is_duplicate((s, "x"), ("The Eiffel Tower was completed in 1889", "y"))
    assert not default_is_duplicate((s, "x"), ("The Eiffel Tower was completed in 1887.", "y"))
    assert default_is_duplicate(("The wall is visible from the Moon", ""), ("The wall is not visible from the Moon", ""))
