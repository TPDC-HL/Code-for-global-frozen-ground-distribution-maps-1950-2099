# Checklist for manuscript code release

- [ ] Confirm code authorship and institutional release permission.
- [ ] Review the parent repository LICENSE and its scope for the added modules.
- [ ] Verify 25 GCMs, tasmax/pr units, grid resolution and SSP experiment files.
- [ ] Confirm KBDI spin-up, 1981–2014 rainfall climatology, initial state, rain-event threshold, and temperature threshold.
- [ ] Run the complete GCM-by-SSP KBDI archive using actual forcing and compare manuscript figures.
- [ ] Verify Frost-number TP threshold 0.55 and elsewhere 0.53 against final Methods.
- [ ] Validate TP mask alignment for FI/TI 0.25-degree grid.
- [ ] Verify R-2 sensor chronology and permissions to release field observations.
- [ ] Obtain electric-field and laboratory triboelectrification observations separately.
- [ ] Check CALM/satellite licenses and citations.
- [ ] Run `PYTHONPATH=src pytest -q tests` on clean clone.
- [ ] Obtain archive DOI (e.g. Zenodo) and update manuscript Code availability only after deposit.
