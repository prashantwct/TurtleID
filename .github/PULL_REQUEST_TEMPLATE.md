## What this changes

<!-- One or two sentences. -->

## Why

<!-- If it corrects reference data, cite the source. -->

## Checks

- [ ] `python -m scripts.validate_db` passes
- [ ] `python -m pytest test_smoke.py -q` passes
- [ ] No coordinates, locality notes, observer names, or API tokens anywhere in the diff
- [ ] `contributions/`, `records/`, `logs/`, `models/*.pt` and `.env` are not staged

## If this touches reference data

- [ ] Diagnostic text is written in my own words, not copied from a handbook
- [ ] Sources cited in the `references` array
- [ ] New species also added to `MATRIX` in `core/morphkey.py`
- [ ] Changes to `wpa_2022` or `cites` verified against a primary source

## If this touches distribution

- [ ] I understand this changes the geographic prior in `core/inference.py` and
      therefore what the model reports for animals found in that state
- [ ] Each added state has a source

## If this touches the confidence pipeline

- [ ] No abstention path removed and no threshold raised in a way that makes the
      tool report a species more often
- [ ] The uncalibrated-model warning still fires
