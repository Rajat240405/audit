# MoES PQ ↔ Parliamentary overlap validation

SUMMARY

MoES corpus root            : E:\audit2\data\.moes-website
Parliamentary corpus root   : E:\audit2\data\parliamentary-qa\rajya-sabha
PQ-titled MoES records      : 100
MoES PQ documents compared  : 188
Exact SHA-256 matches       : 0
Textually near-identical    : 14
Potentially corresponding   : 10
Potentially unique          : 164
Uncomparable                : 0
Total accounted             : 188

Method (deterministic, read-only, no LLM, no network):
- Stage A exact SHA-256: 0 matches / 188 documents (sha-missing on MoES side: 0).
- Text: parliamentary side reuses inline qa.jsonl question+answer text (English); MoES PQ PDFs text-extracted in-memory (PyMuPDF). Records classified via their English document: 100 eng-text records, 0 Hindi-only (uncomparable), 0 without extractable text.
- Normalization: NFC, casefold, page-number lines and 1 corpus-adaptive boilerplate lines removed (>= 0.5 doc-fraction), punctuation/whitespace collapsed.
- Similarity: word 5-gram shingle directional containment |MoES∩RS|/|MoES| (Jaccard reported); candidates prefiltered by title-token coverage (top 8).
- Thresholds: near-identical >= 0.9, corresponding >= 0.5 (PROVISIONAL defaults unless tuned via --calibrate on this corpus).
- Report identity: generated deterministically from corpus inputs; no timestamps.

=== EXACT SHA-256 MATCHES ===

(none)

=== TEXTUALLY NEAR-IDENTICAL ===

- 29586-00-eng [eng] «PARLIAMENT QUESTION: WEATHER FORECASTING AND EARLY WARNINGS»
    parliamentary: rs-271-0530 / 271
    containment 0.958 · jaccard 0.853 · containment 0.958 >= 0.9 vs rs-271-0530 (session 271, date 2026-07-23, Δ1d); title ratio 0.01; shingles 5-gram
- 29586-00-hin [hin] «PARLIAMENT QUESTION: WEATHER FORECASTING AND EARLY WARNINGS»
    parliamentary: rs-271-0530 / 271
    containment 0.958 · jaccard 0.853 · containment 0.958 >= 0.9 vs rs-271-0530 (session 271, date 2026-07-23, Δ1d); title ratio 0.01; shingles 5-gram (record verdict via sibling document)
- 29598-00-eng [eng] «PARLIAMENT QUESTION: IMPLEMENTATION OF MISSION MAUSAM»
    parliamentary: rs-271-0529 / 271
    containment 0.947 · jaccard 0.833 · containment 0.947 >= 0.9 vs rs-271-0529 (session 271, date 2026-07-23, Δ1d); title ratio 0.00; shingles 5-gram
- 29598-00-hin [hin] «PARLIAMENT QUESTION: IMPLEMENTATION OF MISSION MAUSAM»
    parliamentary: rs-271-0529 / 271
    containment 0.947 · jaccard 0.833 · containment 0.947 >= 0.9 vs rs-271-0529 (session 271, date 2026-07-23, Δ1d); title ratio 0.00; shingles 5-gram (record verdict via sibling document)
- 29761-00-eng [eng] «PARLIAMENT QUESTION: OCEAN OBSERVATION NETWORK AND FORECASTING»
    parliamentary: rs-271-1324 / 271
    containment 0.913 · jaccard 0.758 · containment 0.913 >= 0.9 vs rs-271-1324 (session 271, date 2026-07-30, Δ1d); title ratio 0.09; shingles 5-gram
- 29761-00-hin [hin] «PARLIAMENT QUESTION: OCEAN OBSERVATION NETWORK AND FORECASTING»
    parliamentary: rs-271-1324 / 271
    containment 0.913 · jaccard 0.758 · containment 0.913 >= 0.9 vs rs-271-1324 (session 271, date 2026-07-30, Δ1d); title ratio 0.09; shingles 5-gram (record verdict via sibling document)
- 29978-00-eng [eng] «PARLIAMENT QUESTION: STRENGTHENING LONG RANGE FORECASTING CAPABILITIES»
    parliamentary: rs-271-2914 / 271
    containment 0.912 · jaccard 0.717 · containment 0.912 >= 0.9 vs rs-271-2914 (session 271, date 2026-08-13, Δ1d); title ratio 0.00; shingles 5-gram
- 29978-00-hin [hin] «PARLIAMENT QUESTION: STRENGTHENING LONG RANGE FORECASTING CAPABILITIES»
    parliamentary: rs-271-2914 / 271
    containment 0.912 · jaccard 0.717 · containment 0.912 >= 0.9 vs rs-271-2914 (session 271, date 2026-08-13, Δ1d); title ratio 0.00; shingles 5-gram (record verdict via sibling document)
- 29990-00-eng [eng] «PARLIAMENT QUESTION: STRENGTHENING CLIMATE RESILIENCE AND FORECASTING CAPABILITI»
    parliamentary: rs-271-2915 / 271
    containment 0.916 · jaccard 0.759 · containment 0.916 >= 0.9 vs rs-271-2915 (session 271, date 2026-08-13, Δ1d); title ratio 0.01; shingles 5-gram
- 29990-00-hin [hin] «PARLIAMENT QUESTION: STRENGTHENING CLIMATE RESILIENCE AND FORECASTING CAPABILITI»
    parliamentary: rs-271-2915 / 271
    containment 0.916 · jaccard 0.759 · containment 0.916 >= 0.9 vs rs-271-2915 (session 271, date 2026-08-13, Δ1d); title ratio 0.01; shingles 5-gram (record verdict via sibling document)
- 30003-00-eng [eng] «PARLIAMENT QUESTION: INDIGENIOUS WARNING SYSTEMS»
    parliamentary: rs-271-2919 / 271
    containment 0.901 · jaccard 0.742 · containment 0.901 >= 0.9 vs rs-271-2919 (session 271, date 2026-08-13, Δ1d); title ratio 0.00; shingles 5-gram
- 30003-00-hin [hin] «PARLIAMENT QUESTION: INDIGENIOUS WARNING SYSTEMS»
    parliamentary: rs-271-2919 / 271
    containment 0.901 · jaccard 0.742 · containment 0.901 >= 0.9 vs rs-271-2919 (session 271, date 2026-08-13, Δ1d); title ratio 0.00; shingles 5-gram (record verdict via sibling document)
- 30011-00-eng [eng] «PARLIAMENT QUESTION: MISSION MAUSAM»
    parliamentary: rs-271-2916 / 271
    containment 0.913 · jaccard 0.770 · containment 0.913 >= 0.9 vs rs-271-2916 (session 271, date 2026-08-13, Δ1d); title ratio 0.06; shingles 5-gram
- 30011-00-hin [hin] «PARLIAMENT QUESTION: MISSION MAUSAM»
    parliamentary: rs-271-2916 / 271
    containment 0.913 · jaccard 0.770 · containment 0.913 >= 0.9 vs rs-271-2916 (session 271, date 2026-08-13, Δ1d); title ratio 0.06; shingles 5-gram (record verdict via sibling document)

=== POTENTIALLY CORRESPONDING ===

- 29688-00-eng [eng] «PARLIAMENT QUESTION: EXPANSION OF OCEAN OBSERVATION NETWORK»
    parliamentary: rs-271-1324 / 271
    containment 0.521 · jaccard 0.294 · containment 0.521 in [0.5, 0.9) vs rs-271-1324 (session 271, date 2026-07-30, Δ0d); title ratio 0.11
- 29688-00-hin [hin] «PARLIAMENT QUESTION: EXPANSION OF OCEAN OBSERVATION NETWORK»
    parliamentary: rs-271-1324 / 271
    containment 0.521 · jaccard 0.294 · containment 0.521 in [0.5, 0.9) vs rs-271-1324 (session 271, date 2026-07-30, Δ0d); title ratio 0.11 (record verdict via sibling document)
- 29735-00-eng [eng] «PARLIAMENT QUESTION: BRAHMAPUTRA FLOOD FORECASTING AND CLIMATE RESILIENCE MISSIO»
    parliamentary: rs-271-1319 / 271
    containment 0.851 · jaccard 0.667 · containment 0.851 in [0.5, 0.9) vs rs-271-1319 (session 271, date 2026-07-30, Δ1d); title ratio 0.32
- 29735-00-hin [hin] «PARLIAMENT QUESTION: BRAHMAPUTRA FLOOD FORECASTING AND CLIMATE RESILIENCE MISSIO»
    parliamentary: rs-271-1319 / 271
    containment 0.851 · jaccard 0.667 · containment 0.851 in [0.5, 0.9) vs rs-271-1319 (session 271, date 2026-07-30, Δ1d); title ratio 0.32 (record verdict via sibling document)
- 29744-00-eng [eng] «PARLIAMENT QUESTION: IMPACT OF EL NINO ON INDIAN MONSOON»
    parliamentary: rs-271-1323 / 271
    containment 0.796 · jaccard 0.510 · containment 0.796 in [0.5, 0.9) vs rs-271-1323 (session 271, date 2026-07-30, Δ1d); title ratio 0.00
- 29744-00-hin [hin] «PARLIAMENT QUESTION: IMPACT OF EL NINO ON INDIAN MONSOON»
    parliamentary: rs-271-1323 / 271
    containment 0.796 · jaccard 0.510 · containment 0.796 in [0.5, 0.9) vs rs-271-1323 (session 271, date 2026-07-30, Δ1d); title ratio 0.00 (record verdict via sibling document)
- 29752-00-eng [eng] «PARLIAMENT QUESTION: DOPPLER RADAR COVERAGE IN LAHAUL AND SPITI AND KINNAUR»
    parliamentary: rs-271-1320 / 271
    containment 0.754 · jaccard 0.492 · containment 0.754 in [0.5, 0.9) vs rs-271-1320 (session 271, date 2026-07-30, Δ1d); title ratio 0.03
- 29752-00-hin [hin] «PARLIAMENT QUESTION: DOPPLER RADAR COVERAGE IN LAHAUL AND SPITI AND KINNAUR»
    parliamentary: rs-271-1320 / 271
    containment 0.754 · jaccard 0.492 · containment 0.754 in [0.5, 0.9) vs rs-271-1320 (session 271, date 2026-07-30, Δ1d); title ratio 0.03 (record verdict via sibling document)
- 29995-00-eng [eng] «PARLIAMENT QUESTION: AI AND GEOSPATIAL TECHNOLOGIES FOR WEATHER FORECASTING»
    parliamentary: rs-271-2998 / 271
    containment 0.870 · jaccard 0.685 · containment 0.870 in [0.5, 0.9) vs rs-271-2998 (session 271, date 2026-08-13, Δ1d); title ratio 0.01
- 29995-00-hin [hin] «PARLIAMENT QUESTION: AI AND GEOSPATIAL TECHNOLOGIES FOR WEATHER FORECASTING»
    parliamentary: rs-271-2998 / 271
    containment 0.870 · jaccard 0.685 · containment 0.870 in [0.5, 0.9) vs rs-271-2998 (session 271, date 2026-08-13, Δ1d); title ratio 0.01 (record verdict via sibling document)

=== POTENTIALLY UNIQUE MoES PQ DOCUMENTS ===

- 24195-00-eng [eng] «Parliament Question: Monsoon Prediction»
    parliamentary: rs-271-0530 / 271
    containment 0.013 · jaccard 0.004 · best containment 0.013 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-363d))
- 24209-00-eng [eng] «PARLIAMENT QUESTION: PLAN TO IMPROVE WEATHER FORECASTING»
    parliamentary: rs-271-1318 / 271
    containment 0.046 · jaccard 0.013 · best containment 0.046 < 0.5 (best candidate rs-271-1318 (session 271, date 2026-07-30, Δ-370d))
- 24210-00-eng [eng] «PARLIAMENT QUESTION: IMPLEMENTATION OF MISSION MAUSAM IN THE NORTH-EAST REGION»
    parliamentary: rs-271-0529 / 271
    containment 0.016 · jaccard 0.004 · best containment 0.016 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-363d))
- 24211-00-eng [eng] «PARLIAMENT QUESTION: MISSION MAUSAM»
    parliamentary: rs-271-1319 / 271
    containment 0.017 · jaccard 0.010 · best containment 0.017 < 0.5 (best candidate rs-271-1319 (session 271, date 2026-07-30, Δ-370d))
- 24220-00-eng [eng] «PARLIAMENT QUESTION: PROMOTION OF BLUE ECONOMY»
    parliamentary: rs-271-2916 / 271
    containment 0.007 · jaccard 0.004 · best containment 0.007 < 0.5 (best candidate rs-271-2916 (session 271, date 2026-08-13, Δ-384d))
- 24224-00-eng [eng] «PARLIAMENT QUESTION: Doppler Radar Station at Balasore»
    parliamentary: rs-271-1320 / 271
    containment 0.012 · jaccard 0.005 · best containment 0.012 < 0.5 (best candidate rs-271-1320 (session 271, date 2026-07-30, Δ-370d))
- 24227-00-eng [eng] «PARLIAMENT QUESTION: EL NINO EFFECT ON MONSOON AND RAINFALL»
    parliamentary: rs-271-2914 / 271
    containment 0.024 · jaccard 0.006 · best containment 0.024 < 0.5 (best candidate rs-271-2914 (session 271, date 2026-08-13, Δ-384d))
- 24229-00-eng [eng] «PARLIAMENT QUESTION: VIKSIT BHARAT»
    parliamentary: -
    containment 0.000 · jaccard 0.000 · best containment 0.000 < 0.5 (best candidate none)
- 24234-00-eng [eng] «Parliament Question: Mission Mausam To Boost The Radar Network»
    parliamentary: rs-271-1319 / 271
    containment 0.028 · jaccard 0.011 · best containment 0.028 < 0.5 (best candidate rs-271-1319 (session 271, date 2026-07-30, Δ-370d))
- 24240-00-eng [eng] «Parliament Question: Fourth Global Coral Bleaching Event»
    parliamentary: rs-271-2921 / 271
    containment 0.012 · jaccard 0.007 · best containment 0.012 < 0.5 (best candidate rs-271-2921 (session 271, date 2026-08-13, Δ-384d))
- 26811-00-eng [eng] «PARLIAMENT QUESTION: ADVERSE EFFECTS OF CLIMATE CHANGE ON HIMALAYAS»
    parliamentary: rs-271-2916 / 271
    containment 0.005 · jaccard 0.002 · best containment 0.005 < 0.5 (best candidate rs-271-2916 (session 271, date 2026-08-13, Δ-252d))
- 26811-00-hin [hin] «PARLIAMENT QUESTION: ADVERSE EFFECTS OF CLIMATE CHANGE ON HIMALAYAS»
    parliamentary: rs-271-2916 / 271
    containment 0.005 · jaccard 0.002 · best containment 0.005 < 0.5 (best candidate rs-271-2916 (session 271, date 2026-08-13, Δ-252d)) (record verdict via sibling document)
- 26817-00-eng [eng] «PARLIAMENT QUESTION: OPERATIONAL FORECASTING AND MONITORING MECHANISMS»
    parliamentary: rs-271-0529 / 271
    containment 0.002 · jaccard 0.001 · best containment 0.002 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-231d))
- 26817-00-hin [hin] «PARLIAMENT QUESTION: OPERATIONAL FORECASTING AND MONITORING MECHANISMS»
    parliamentary: rs-271-0529 / 271
    containment 0.002 · jaccard 0.001 · best containment 0.002 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-231d)) (record verdict via sibling document)
- 26826-00-eng [eng] «PARLIAMENT QUESTION: MAPPING OF HEATWAVES»
    parliamentary: rs-271-0274 / 271
    containment 0.006 · jaccard 0.001 · best containment 0.006 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-252d))
- 26826-00-hin [hin] «PARLIAMENT QUESTION: MAPPING OF HEATWAVES»
    parliamentary: rs-271-0274 / 271
    containment 0.006 · jaccard 0.001 · best containment 0.006 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-252d)) (record verdict via sibling document)
- 26829-00-eng [eng] «PARLIAMENT QUESTION: WINTER FORECAST BY IMD»
    parliamentary: rs-271-0274 / 271
    containment 0.003 · jaccard 0.001 · best containment 0.003 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-252d))
- 26829-00-hin [hin] «PARLIAMENT QUESTION: WINTER FORECAST BY IMD»
    parliamentary: rs-271-0274 / 271
    containment 0.003 · jaccard 0.001 · best containment 0.003 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-252d)) (record verdict via sibling document)
- 27061-00-eng [eng] «PARLIAMENT QUESTION: OPERATIONAL FORECASTING AND MONITORING MECHANISMS»
    parliamentary: rs-271-0529 / 271
    containment 0.002 · jaccard 0.001 · best containment 0.002 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-230d))
- 27061-00-hin [hin] «PARLIAMENT QUESTION: OPERATIONAL FORECASTING AND MONITORING MECHANISMS»
    parliamentary: rs-271-0529 / 271
    containment 0.002 · jaccard 0.001 · best containment 0.002 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-230d)) (record verdict via sibling document)
- 27067-00-eng [eng] «PARLIAMENT QUESTION: RESEARCH PROJECTS UNDER PRITHVI SCHEME»
    parliamentary: -
    containment 0.000 · jaccard 0.000 · best containment 0.000 < 0.5 (best candidate none)
- 27067-00-hin [hin] «PARLIAMENT QUESTION: RESEARCH PROJECTS UNDER PRITHVI SCHEME»
    parliamentary: -
    containment 0.000 · jaccard 0.000 · best containment 0.000 < 0.5 (best candidate none) (record verdict via sibling document)
- 27073-00-eng [eng] «PARLIAMENT QUESTION: WEATHER FORECASTS BY REGIONAL METEOROLOGICAL CENTRE (RMC) K»
    parliamentary: rs-271-2915 / 271
    containment 0.005 · jaccard 0.002 · best containment 0.005 < 0.5 (best candidate rs-271-2915 (session 271, date 2026-08-13, Δ-251d))
- 27073-00-hin [hin] «PARLIAMENT QUESTION: WEATHER FORECASTS BY REGIONAL METEOROLOGICAL CENTRE (RMC) K»
    parliamentary: rs-271-2915 / 271
    containment 0.005 · jaccard 0.002 · best containment 0.005 < 0.5 (best candidate rs-271-2915 (session 271, date 2026-08-13, Δ-251d)) (record verdict via sibling document)
- 27080-00-eng [eng] «PARLIAMENT QUESTION: MARINE MINING AND WEATHER FORECASTING»
    parliamentary: rs-271-0529 / 271
    containment 0.315 · jaccard 0.069 · best containment 0.315 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-230d))
- 27080-00-hin [hin] «PARLIAMENT QUESTION: MARINE MINING AND WEATHER FORECASTING»
    parliamentary: rs-271-0529 / 271
    containment 0.315 · jaccard 0.069 · best containment 0.315 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-230d)) (record verdict via sibling document)
- 27086-00-eng [eng] «PARLIAMENT QUESTION: DEEP-SEA EXPLORATION CONTRACTS»
    parliamentary: rs-271-1322 / 271
    containment 0.092 · jaccard 0.030 · best containment 0.092 < 0.5 (best candidate rs-271-1322 (session 271, date 2026-07-30, Δ-237d))
- 27086-00-hin [hin] «PARLIAMENT QUESTION: DEEP-SEA EXPLORATION CONTRACTS»
    parliamentary: rs-271-1322 / 271
    containment 0.092 · jaccard 0.030 · best containment 0.092 < 0.5 (best candidate rs-271-1322 (session 271, date 2026-07-30, Δ-237d)) (record verdict via sibling document)
- 27094-00-eng [eng] «PARLIAMENT QUESTION: DEEP SEA EXPLORATION FOR MINERALS, OIL AND NATURAL GAS»
    parliamentary: rs-271-1322 / 271
    containment 0.088 · jaccard 0.035 · best containment 0.088 < 0.5 (best candidate rs-271-1322 (session 271, date 2026-07-30, Δ-237d))
- 27094-00-hin [hin] «PARLIAMENT QUESTION: DEEP SEA EXPLORATION FOR MINERALS, OIL AND NATURAL GAS»
    parliamentary: rs-271-1322 / 271
    containment 0.088 · jaccard 0.035 · best containment 0.088 < 0.5 (best candidate rs-271-1322 (session 271, date 2026-07-30, Δ-237d)) (record verdict via sibling document)
- 27099-00-eng [eng] «PARLIAMENT QUESTION: Coastline of the Country»
    parliamentary: -
    containment 0.000 · jaccard 0.000 · best containment 0.000 < 0.5 (best candidate none)
- 27099-00-hin [hin] «PARLIAMENT QUESTION: Coastline of the Country»
    parliamentary: -
    containment 0.000 · jaccard 0.000 · best containment 0.000 < 0.5 (best candidate none) (record verdict via sibling document)
- 27130-00-eng [eng] «PARLIAMENT QUESTION: ADVANCED COMPUTER SIMULATION MODELS TO IMPROVE LOCALISED WE»
    parliamentary: rs-271-0530 / 271
    containment 0.008 · jaccard 0.003 · best containment 0.008 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-224d))
- 27130-00-hin [hin] «PARLIAMENT QUESTION: ADVANCED COMPUTER SIMULATION MODELS TO IMPROVE LOCALISED WE»
    parliamentary: rs-271-0530 / 271
    containment 0.008 · jaccard 0.003 · best containment 0.008 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-224d)) (record verdict via sibling document)
- 27136-00-eng [eng] «PARLIAMENT QUESTION: OFFSHORE DEEP-SEA MINING IN KERALA»
    parliamentary: rs-271-0274 / 271
    containment 0.002 · jaccard 0.001 · best containment 0.002 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-245d))
- 27136-00-hin [hin] «PARLIAMENT QUESTION: OFFSHORE DEEP-SEA MINING IN KERALA»
    parliamentary: rs-271-0274 / 271
    containment 0.002 · jaccard 0.001 · best containment 0.002 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-245d)) (record verdict via sibling document)
- 27145-00-eng [eng] «PARLIAMENT QUESTION: MAITRI-2 STATION»
    parliamentary: rs-271-1320 / 271
    containment 0.009 · jaccard 0.002 · best containment 0.009 < 0.5 (best candidate rs-271-1320 (session 271, date 2026-07-30, Δ-231d))
- 27145-00-hin [hin] «PARLIAMENT QUESTION: MAITRI-2 STATION»
    parliamentary: rs-271-1320 / 271
    containment 0.009 · jaccard 0.002 · best containment 0.009 < 0.5 (best candidate rs-271-1320 (session 271, date 2026-07-30, Δ-231d)) (record verdict via sibling document)
- 27151-00-eng [eng] «PARLIAMENT QUESTION: Vulnerability Of Coastal Region to Flood and Sea Level Rise»
    parliamentary: rs-271-1319 / 271
    containment 0.006 · jaccard 0.004 · best containment 0.006 < 0.5 (best candidate rs-271-1319 (session 271, date 2026-07-30, Δ-231d))
- 27151-00-hin [hin] «PARLIAMENT QUESTION: Vulnerability Of Coastal Region to Flood and Sea Level Rise»
    parliamentary: rs-271-1319 / 271
    containment 0.006 · jaccard 0.004 · best containment 0.006 < 0.5 (best candidate rs-271-1319 (session 271, date 2026-07-30, Δ-231d)) (record verdict via sibling document)
- 27154-00-eng [eng] «PARLIAMENT QUESTION: DEEP OCEAN MISSION»
    parliamentary: rs-271-1322 / 271
    containment 0.165 · jaccard 0.050 · best containment 0.165 < 0.5 (best candidate rs-271-1322 (session 271, date 2026-07-30, Δ-231d))
- 27154-00-hin [hin] «PARLIAMENT QUESTION: DEEP OCEAN MISSION»
    parliamentary: rs-271-1322 / 271
    containment 0.165 · jaccard 0.050 · best containment 0.165 < 0.5 (best candidate rs-271-1322 (session 271, date 2026-07-30, Δ-231d)) (record verdict via sibling document)
- 27160-00-eng [eng] «PARLIAMENT QUESTION: WEATHER AND CLIMATE SERVICES»
    parliamentary: rs-271-0529 / 271
    containment 0.054 · jaccard 0.017 · best containment 0.054 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-223d))
- 27160-00-hin [hin] «PARLIAMENT QUESTION: WEATHER AND CLIMATE SERVICES»
    parliamentary: rs-271-0529 / 271
    containment 0.054 · jaccard 0.017 · best containment 0.054 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-223d)) (record verdict via sibling document)
- 27166-00-eng [eng] «PARLIAMENT QUESTION: NATIONAL CLIMATE SERVICES»
    parliamentary: rs-271-2915 / 271
    containment 0.007 · jaccard 0.003 · best containment 0.007 < 0.5 (best candidate rs-271-2915 (session 271, date 2026-08-13, Δ-244d))
- 27166-00-hin [hin] «PARLIAMENT QUESTION: NATIONAL CLIMATE SERVICES»
    parliamentary: rs-271-2915 / 271
    containment 0.007 · jaccard 0.003 · best containment 0.007 < 0.5 (best candidate rs-271-2915 (session 271, date 2026-08-13, Δ-244d)) (record verdict via sibling document)
- 27172-00-eng [eng] «PARLIAMENT QUESTION: UNPREDICTABLE WEATHER PATTERNS»
    parliamentary: rs-271-0529 / 271
    containment 0.025 · jaccard 0.013 · best containment 0.025 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-223d))
- 27172-00-hin [hin] «PARLIAMENT QUESTION: UNPREDICTABLE WEATHER PATTERNS»
    parliamentary: rs-271-0529 / 271
    containment 0.025 · jaccard 0.013 · best containment 0.025 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-223d)) (record verdict via sibling document)
- 27188-00-eng [eng] «PARLIAMENT QUESTION: HIGH-RISK SEISMIC CATEGORISATION OF THE HIMALAYAN REGION»
    parliamentary: rs-271-0274 / 271
    containment 0.010 · jaccard 0.002 · best containment 0.010 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-238d))
- 27188-00-hin [hin] «PARLIAMENT QUESTION: HIGH-RISK SEISMIC CATEGORISATION OF THE HIMALAYAN REGION»
    parliamentary: rs-271-0274 / 271
    containment 0.010 · jaccard 0.002 · best containment 0.010 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-238d)) (record verdict via sibling document)
- 27195-00-eng [eng] «PARLIAMENT QUESTION: OCEAN MINING AND TECHNOLOGY UPGRADATION»
    parliamentary: rs-271-1322 / 271
    containment 0.127 · jaccard 0.026 · best containment 0.127 < 0.5 (best candidate rs-271-1322 (session 271, date 2026-07-30, Δ-224d))
- 27195-00-hin [hin] «PARLIAMENT QUESTION: OCEAN MINING AND TECHNOLOGY UPGRADATION»
    parliamentary: rs-271-1322 / 271
    containment 0.127 · jaccard 0.026 · best containment 0.127 < 0.5 (best candidate rs-271-1322 (session 271, date 2026-07-30, Δ-224d)) (record verdict via sibling document)
- 27201-00-eng [eng] «PARLIAMENT QUESTION: MONSOON WARNING SYSTEM»
    parliamentary: rs-271-0530 / 271
    containment 0.021 · jaccard 0.009 · best containment 0.021 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-217d))
- 27201-00-hin [hin] «PARLIAMENT QUESTION: MONSOON WARNING SYSTEM»
    parliamentary: rs-271-0530 / 271
    containment 0.021 · jaccard 0.009 · best containment 0.021 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-217d)) (record verdict via sibling document)
- 27207-00-eng [eng] «PARLIAMENT QUESTION: FORECASTING OF HEAVY RAINS AND LANDSLIDES»
    parliamentary: rs-271-1318 / 271
    containment 0.005 · jaccard 0.003 · best containment 0.005 < 0.5 (best candidate rs-271-1318 (session 271, date 2026-07-30, Δ-224d))
- 27207-00-hin [hin] «PARLIAMENT QUESTION: FORECASTING OF HEAVY RAINS AND LANDSLIDES»
    parliamentary: rs-271-1318 / 271
    containment 0.005 · jaccard 0.003 · best containment 0.005 < 0.5 (best candidate rs-271-1318 (session 271, date 2026-07-30, Δ-224d)) (record verdict via sibling document)
- 27212-00-eng [eng] «PARLIAMENT QUESTION: RISING THREATS OF CLIMATE CHANGE»
    parliamentary: rs-271-1319 / 271
    containment 0.019 · jaccard 0.015 · best containment 0.019 < 0.5 (best candidate rs-271-1319 (session 271, date 2026-07-30, Δ-224d))
- 27212-00-hin [hin] «PARLIAMENT QUESTION: RISING THREATS OF CLIMATE CHANGE»
    parliamentary: rs-271-1319 / 271
    containment 0.019 · jaccard 0.015 · best containment 0.019 < 0.5 (best candidate rs-271-1319 (session 271, date 2026-07-30, Δ-224d)) (record verdict via sibling document)
- 27219-00-eng [eng] «PARLIAMENT QUESTION: MONITORING SYSTEM FOR RISK PRONE AREAS»
    parliamentary: rs-271-0529 / 271
    containment 0.007 · jaccard 0.003 · best containment 0.007 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-216d))
- 27219-00-hin [hin] «PARLIAMENT QUESTION: MONITORING SYSTEM FOR RISK PRONE AREAS»
    parliamentary: rs-271-0529 / 271
    containment 0.007 · jaccard 0.003 · best containment 0.007 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-216d)) (record verdict via sibling document)
- 27227-00-eng [eng] «PARLIAMENT QUESTION: IMPROVING CLIMATE AND WEATHER SERVICES»
    parliamentary: rs-271-0530 / 271
    containment 0.010 · jaccard 0.004 · best containment 0.010 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-216d))
- 27227-00-hin [hin] «PARLIAMENT QUESTION: IMPROVING CLIMATE AND WEATHER SERVICES»
    parliamentary: rs-271-0530 / 271
    containment 0.010 · jaccard 0.004 · best containment 0.010 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-216d)) (record verdict via sibling document)
- 27234-00-eng [eng] «PARLIAMENT QUESTION: ACCURACY OF FORECAST»
    parliamentary: rs-271-2914 / 271
    containment 0.001 · jaccard 0.001 · best containment 0.001 < 0.5 (best candidate rs-271-2914 (session 271, date 2026-08-13, Δ-237d))
- 27234-00-hin [hin] «PARLIAMENT QUESTION: ACCURACY OF FORECAST»
    parliamentary: rs-271-2914 / 271
    containment 0.001 · jaccard 0.001 · best containment 0.001 < 0.5 (best candidate rs-271-2914 (session 271, date 2026-08-13, Δ-237d)) (record verdict via sibling document)
- 27238-00-eng [eng] «PARLIAMENT QUESTION: MISSION MAUSAM»
    parliamentary: rs-271-0529 / 271
    containment 0.172 · jaccard 0.042 · best containment 0.172 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-216d))
- 27238-00-hin [hin] «PARLIAMENT QUESTION: MISSION MAUSAM»
    parliamentary: rs-271-0529 / 271
    containment 0.172 · jaccard 0.042 · best containment 0.172 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-216d)) (record verdict via sibling document)
- 27419-00-eng [eng] «PARLIAMENT QUESTION: ADVANCEMENTS IN WEATHER FORECASTING»
    parliamentary: rs-271-0529 / 271
    containment 0.101 · jaccard 0.064 · best containment 0.101 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-174d))
- 27419-00-hin [hin] «PARLIAMENT QUESTION: ADVANCEMENTS IN WEATHER FORECASTING»
    parliamentary: rs-271-0529 / 271
    containment 0.101 · jaccard 0.064 · best containment 0.101 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-174d)) (record verdict via sibling document)
- 27428-00-eng [eng] «PARLIAMENT QUESTION:  UPGRADATION IN OBSERVATION NETWORK»
    parliamentary: rs-271-0529 / 271
    containment 0.013 · jaccard 0.004 · best containment 0.013 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-174d))
- 27428-00-hin [hin] «PARLIAMENT QUESTION:  UPGRADATION IN OBSERVATION NETWORK»
    parliamentary: rs-271-0529 / 271
    containment 0.013 · jaccard 0.004 · best containment 0.013 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-174d)) (record verdict via sibling document)
- 27437-00-eng [eng] «PARLIAMENT QUESTION: FORECAST SYSTEM»
    parliamentary: rs-271-0274 / 271
    containment 0.006 · jaccard 0.001 · best containment 0.006 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-195d))
- 27437-00-hin [hin] «PARLIAMENT QUESTION: FORECAST SYSTEM»
    parliamentary: rs-271-0274 / 271
    containment 0.006 · jaccard 0.001 · best containment 0.006 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-195d)) (record verdict via sibling document)
- 27689-00-eng [eng] «PARLIAMENT QUESTION: STUDIES TO ASSESS THE IMPACT OF EXTREME WEATHER CONDITIONS»
    parliamentary: rs-271-0529 / 271
    containment 0.063 · jaccard 0.024 · best containment 0.063 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-161d))
- 27689-00-hin [hin] «PARLIAMENT QUESTION: STUDIES TO ASSESS THE IMPACT OF EXTREME WEATHER CONDITIONS»
    parliamentary: rs-271-0529 / 271
    containment 0.063 · jaccard 0.024 · best containment 0.063 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-161d)) (record verdict via sibling document)
- 27698-00-eng [eng] «PARLIAMENT QUESTION: ACCURATE WEATHER FORECASTING»
    parliamentary: rs-271-0529 / 271
    containment 0.063 · jaccard 0.024 · best containment 0.063 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-161d))
- 27698-00-hin [hin] «PARLIAMENT QUESTION: ACCURATE WEATHER FORECASTING»
    parliamentary: rs-271-0529 / 271
    containment 0.063 · jaccard 0.024 · best containment 0.063 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-161d)) (record verdict via sibling document)
- 27708-00-eng [eng] «PARLIAMENT QUESTION: PRITHVI VIGYAN SCHEME»
    parliamentary: -
    containment 0.000 · jaccard 0.000 · best containment 0.000 < 0.5 (best candidate none)
- 27708-00-hin [hin] «PARLIAMENT QUESTION: PRITHVI VIGYAN SCHEME»
    parliamentary: -
    containment 0.000 · jaccard 0.000 · best containment 0.000 < 0.5 (best candidate none) (record verdict via sibling document)
- 27714-00-eng [eng] «PARLIAMENT QUESTION: MONSOON FORECASTING AND CLIMATE RESILIENCE IN ASSAM»
    parliamentary: rs-271-2916 / 271
    containment 0.009 · jaccard 0.004 · best containment 0.009 < 0.5 (best candidate rs-271-2916 (session 271, date 2026-08-13, Δ-182d))
- 27714-00-hin [hin] «PARLIAMENT QUESTION: MONSOON FORECASTING AND CLIMATE RESILIENCE IN ASSAM»
    parliamentary: rs-271-2916 / 271
    containment 0.009 · jaccard 0.004 · best containment 0.009 < 0.5 (best candidate rs-271-2916 (session 271, date 2026-08-13, Δ-182d)) (record verdict via sibling document)
- 27724-00-eng [eng] «PARLIAMENT QUESTION: ACCURACY OF WEATHER FORECASTS»
    parliamentary: rs-271-0529 / 271
    containment 0.025 · jaccard 0.007 · best containment 0.025 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-161d))
- 27724-00-hin [hin] «PARLIAMENT QUESTION: ACCURACY OF WEATHER FORECASTS»
    parliamentary: rs-271-0529 / 271
    containment 0.025 · jaccard 0.007 · best containment 0.025 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-161d)) (record verdict via sibling document)
- 27730-00-eng [eng] «PARLIAMENT QUESTION: Studies for Coastal Erosion»
    parliamentary: rs-271-2915 / 271
    containment 0.009 · jaccard 0.003 · best containment 0.009 < 0.5 (best candidate rs-271-2915 (session 271, date 2026-08-13, Δ-182d))
- 27730-00-hin [hin] «PARLIAMENT QUESTION: Studies for Coastal Erosion»
    parliamentary: rs-271-2915 / 271
    containment 0.009 · jaccard 0.003 · best containment 0.009 < 0.5 (best candidate rs-271-2915 (session 271, date 2026-08-13, Δ-182d)) (record verdict via sibling document)
- 27860-00-eng [eng] «PARLIAMENT QUESTION: Performance of early warning systems»
    parliamentary: rs-271-0274 / 271
    containment 0.017 · jaccard 0.008 · best containment 0.017 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-181d))
- 27860-00-hin [hin] «PARLIAMENT QUESTION: Performance of early warning systems»
    parliamentary: rs-271-0274 / 271
    containment 0.017 · jaccard 0.008 · best containment 0.017 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-181d)) (record verdict via sibling document)
- 27873-00-eng [eng] «PARLIAMENT QUESTION: INSTALLATION OF WEATHER STATIONS»
    parliamentary: rs-271-0274 / 271
    containment 0.012 · jaccard 0.003 · best containment 0.012 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-181d))
- 27873-00-hin [hin] «PARLIAMENT QUESTION: INSTALLATION OF WEATHER STATIONS»
    parliamentary: rs-271-0274 / 271
    containment 0.012 · jaccard 0.003 · best containment 0.012 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-181d)) (record verdict via sibling document)
- 27879-00-eng [eng] «PARLIAMENT QUESTION: IMPROVING WEATHER FORECASTING CAPABILITIES»
    parliamentary: rs-271-0530 / 271
    containment 0.132 · jaccard 0.065 · best containment 0.132 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-160d))
- 27879-00-hin [hin] «PARLIAMENT QUESTION: IMPROVING WEATHER FORECASTING CAPABILITIES»
    parliamentary: rs-271-0530 / 271
    containment 0.132 · jaccard 0.065 · best containment 0.132 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-160d)) (record verdict via sibling document)
- 27886-00-eng [eng] «PARLIAMENT QUESTION: ACCURACY OF WEATHER FORECASTING SYSTEMS»
    parliamentary: rs-271-0529 / 271
    containment 0.029 · jaccard 0.012 · best containment 0.029 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-160d))
- 27893-00-eng [eng] «PARLIAMENT QUESTION: STRENGTHEING RADAR INFRASTRUCTURE IN KERALA»
    parliamentary: rs-271-2131 / 271
    containment 0.021 · jaccard 0.006 · best containment 0.021 < 0.5 (best candidate rs-271-2131 (session 271, date 2026-08-06, Δ-174d))
- 27893-00-hin [hin] «PARLIAMENT QUESTION: STRENGTHEING RADAR INFRASTRUCTURE IN KERALA»
    parliamentary: rs-271-2131 / 271
    containment 0.021 · jaccard 0.006 · best containment 0.021 < 0.5 (best candidate rs-271-2131 (session 271, date 2026-08-06, Δ-174d)) (record verdict via sibling document)
- 27917-00-eng [eng] «PARLIAMENT QUESTION: MONSOON FORECAST ACCURACY AND IMPROVEMENTS»
    parliamentary: rs-271-1318 / 271
    containment 0.014 · jaccard 0.006 · best containment 0.014 < 0.5 (best candidate rs-271-1318 (session 271, date 2026-07-30, Δ-167d))
- 27917-00-hin [hin] «PARLIAMENT QUESTION: MONSOON FORECAST ACCURACY AND IMPROVEMENTS»
    parliamentary: rs-271-1318 / 271
    containment 0.014 · jaccard 0.006 · best containment 0.014 < 0.5 (best candidate rs-271-1318 (session 271, date 2026-07-30, Δ-167d)) (record verdict via sibling document)
- 28100-00-eng [eng] «PARLIAMENT QUESTION: Mission Mausam in Odisha»
    parliamentary: rs-271-0274 / 271
    containment 0.011 · jaccard 0.004 · best containment 0.011 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-154d))
- 28100-00-hin [hin] «PARLIAMENT QUESTION: Mission Mausam in Odisha»
    parliamentary: rs-271-0274 / 271
    containment 0.011 · jaccard 0.004 · best containment 0.011 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-154d)) (record verdict via sibling document)
- 28112-00-eng [eng] «PARLIAMENT QUESTIONS: MAJOR IMPROVEMENTS IN WEATHER FORECASTING»
    parliamentary: rs-271-0529 / 271
    containment 0.034 · jaccard 0.020 · best containment 0.034 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-133d))
- 28112-00-hin [hin] «PARLIAMENT QUESTIONS: MAJOR IMPROVEMENTS IN WEATHER FORECASTING»
    parliamentary: rs-271-0529 / 271
    containment 0.034 · jaccard 0.020 · best containment 0.034 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-133d)) (record verdict via sibling document)
- 28117-00-eng [eng] «PARLIAMENT QUESTION: EARLY WARNING SYSTEMS FOR FLOODS AND CYCLONES»
    parliamentary: rs-271-0274 / 271
    containment 0.008 · jaccard 0.004 · best containment 0.008 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-154d))
- 28117-00-hin [hin] «PARLIAMENT QUESTION: EARLY WARNING SYSTEMS FOR FLOODS AND CYCLONES»
    parliamentary: rs-271-0274 / 271
    containment 0.008 · jaccard 0.004 · best containment 0.008 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-154d)) (record verdict via sibling document)
- 28125-00-eng [eng] «PARLIAMENT QUESTION: Mission Mausam»
    parliamentary: rs-271-0529 / 271
    containment 0.172 · jaccard 0.042 · best containment 0.172 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-133d))
- 28125-00-hin [hin] «PARLIAMENT QUESTION: Mission Mausam»
    parliamentary: rs-271-0529 / 271
    containment 0.172 · jaccard 0.042 · best containment 0.172 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-133d)) (record verdict via sibling document)
- 28132-00-eng [eng] «PARLIAMENT QUESTION: Research and Training programmes»
    parliamentary: rs-271-2998 / 271
    containment 0.008 · jaccard 0.005 · best containment 0.008 < 0.5 (best candidate rs-271-2998 (session 271, date 2026-08-13, Δ-154d))
- 28132-00-hin [hin] «PARLIAMENT QUESTION: Research and Training programmes»
    parliamentary: rs-271-2998 / 271
    containment 0.008 · jaccard 0.005 · best containment 0.008 < 0.5 (best candidate rs-271-2998 (session 271, date 2026-08-13, Δ-154d)) (record verdict via sibling document)
- 28144-00-eng [eng] «PARLIAMENT QUESTION: ACCURATE WEATHER FORECASTING»
    parliamentary: rs-271-0529 / 271
    containment 0.011 · jaccard 0.006 · best containment 0.011 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-133d))
- 28144-00-hin [hin] «PARLIAMENT QUESTION: ACCURATE WEATHER FORECASTING»
    parliamentary: rs-271-0529 / 271
    containment 0.011 · jaccard 0.006 · best containment 0.011 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-133d)) (record verdict via sibling document)
- 28149-00-eng [eng] «PARLIAMENT QUESTION: Implementation of Mission Mausam»
    parliamentary: rs-271-0530 / 271
    containment 0.028 · jaccard 0.008 · best containment 0.028 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-133d))
- 28149-00-hin [hin] «PARLIAMENT QUESTION: Implementation of Mission Mausam»
    parliamentary: rs-271-0530 / 271
    containment 0.028 · jaccard 0.008 · best containment 0.028 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-133d)) (record verdict via sibling document)
- 28159-00-eng [eng] «PARLIAMENT QUESTION: AI IN WEATHER FORECASTING»
    parliamentary: rs-271-1318 / 271
    containment 0.112 · jaccard 0.051 · best containment 0.112 < 0.5 (best candidate rs-271-1318 (session 271, date 2026-07-30, Δ-140d))
- 28159-00-hin [hin] «PARLIAMENT QUESTION: AI IN WEATHER FORECASTING»
    parliamentary: rs-271-1318 / 271
    containment 0.112 · jaccard 0.051 · best containment 0.112 < 0.5 (best candidate rs-271-1318 (session 271, date 2026-07-30, Δ-140d)) (record verdict via sibling document)
- 28172-00-eng [eng] «PARLIAMENT QUESTION: DEEP-SEA MINING»
    parliamentary: rs-271-1324 / 271
    containment 0.023 · jaccard 0.010 · best containment 0.023 < 0.5 (best candidate rs-271-1324 (session 271, date 2026-07-30, Δ-139d))
- 28172-00-hin [hin] «PARLIAMENT QUESTION: DEEP-SEA MINING»
    parliamentary: rs-271-1324 / 271
    containment 0.023 · jaccard 0.010 · best containment 0.023 < 0.5 (best candidate rs-271-1324 (session 271, date 2026-07-30, Δ-139d)) (record verdict via sibling document)
- 28184-00-eng [eng] «PARLIAMENT QUESTION: EXTREME WEATHER EVENTS»
    parliamentary: rs-271-0529 / 271
    containment 0.028 · jaccard 0.013 · best containment 0.028 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-132d))
- 28184-00-hin [hin] «PARLIAMENT QUESTION: EXTREME WEATHER EVENTS»
    parliamentary: rs-271-0529 / 271
    containment 0.028 · jaccard 0.013 · best containment 0.028 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-132d)) (record verdict via sibling document)
- 28194-00-eng [eng] «PARLIAMENT QUESTION: STATUS OF IMPLEMENTATION OF MISSION MAUSAM»
    parliamentary: rs-271-0530 / 271
    containment 0.026 · jaccard 0.008 · best containment 0.026 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-132d))
- 28194-00-hin [hin] «PARLIAMENT QUESTION: STATUS OF IMPLEMENTATION OF MISSION MAUSAM»
    parliamentary: rs-271-0530 / 271
    containment 0.026 · jaccard 0.008 · best containment 0.026 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-132d)) (record verdict via sibling document)
- 28239-00-eng [eng] «PARLIAMENT QUESTION: Mission Mausam»
    parliamentary: rs-271-0530 / 271
    containment 0.017 · jaccard 0.006 · best containment 0.017 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-126d))
- 28239-00-hin [hin] «PARLIAMENT QUESTION: Mission Mausam»
    parliamentary: rs-271-0530 / 271
    containment 0.017 · jaccard 0.006 · best containment 0.017 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-126d)) (record verdict via sibling document)
- 28247-00-eng [eng] «PARLIAMENT QUESTION: Automatic Weather Stations»
    parliamentary: rs-271-0274 / 271
    containment 0.010 · jaccard 0.002 · best containment 0.010 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-147d))
- 28247-00-hin [hin] «PARLIAMENT QUESTION: Automatic Weather Stations»
    parliamentary: rs-271-0274 / 271
    containment 0.010 · jaccard 0.002 · best containment 0.010 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-147d)) (record verdict via sibling document)
- 28256-00-eng [eng] «PARLIAMENT QUESTION: REAL-TIME WEATHER UPDATES»
    parliamentary: rs-271-0530 / 271
    containment 0.125 · jaccard 0.062 · best containment 0.125 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-126d))
- 28256-00-hin [hin] «PARLIAMENT QUESTION: REAL-TIME WEATHER UPDATES»
    parliamentary: rs-271-0530 / 271
    containment 0.125 · jaccard 0.062 · best containment 0.125 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ-126d)) (record verdict via sibling document)
- 28265-00-eng [eng] «PARLIAMENT QUESTION: STUDIES FOR CLIMATE CHANGE»
    parliamentary: rs-271-2915 / 271
    containment 0.008 · jaccard 0.004 · best containment 0.008 < 0.5 (best candidate rs-271-2915 (session 271, date 2026-08-13, Δ-147d))
- 28265-00-hin [hin] «PARLIAMENT QUESTION: STUDIES FOR CLIMATE CHANGE»
    parliamentary: rs-271-2915 / 271
    containment 0.008 · jaccard 0.004 · best containment 0.008 < 0.5 (best candidate rs-271-2915 (session 271, date 2026-08-13, Δ-147d)) (record verdict via sibling document)
- 28339-00-eng [eng] «PARLIAMENT QUESTION: ADVANCED METEOROLOGICAL INFRASTRUCTURE»
    parliamentary: rs-271-0529 / 271
    containment 0.012 · jaccard 0.006 · best containment 0.012 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-118d))
- 28339-00-hin [hin] «PARLIAMENT QUESTION: ADVANCED METEOROLOGICAL INFRASTRUCTURE»
    parliamentary: rs-271-0529 / 271
    containment 0.012 · jaccard 0.006 · best containment 0.012 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ-118d)) (record verdict via sibling document)
- 28346-00-eng [eng] «PARLIAMENT QUESTION: FINANCIAL LAPSES IN INSTITUTIONS UNDER THE MINISTRY OF  EAR»
    parliamentary: rs-271-0274 / 271
    containment 0.024 · jaccard 0.007 · best containment 0.024 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-139d))
- 28346-00-hin [hin] «PARLIAMENT QUESTION: FINANCIAL LAPSES IN INSTITUTIONS UNDER THE MINISTRY OF  EAR»
    parliamentary: rs-271-0274 / 271
    containment 0.024 · jaccard 0.007 · best containment 0.024 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-139d)) (record verdict via sibling document)
- 28355-00-eng [eng] «PARLIAMENT QUESTION: Earthquake Vulnerability and Steps taken to Mitigate the Ri»
    parliamentary: rs-271-0274 / 271
    containment 0.023 · jaccard 0.005 · best containment 0.023 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-139d))
- 28355-00-hin [hin] «PARLIAMENT QUESTION: Earthquake Vulnerability and Steps taken to Mitigate the Ri»
    parliamentary: rs-271-0274 / 271
    containment 0.023 · jaccard 0.005 · best containment 0.023 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-139d)) (record verdict via sibling document)
- 29536-00-eng [eng] «PARLIAMENT QUESTION: Earthquake Risk in Kangra-Chamba-Dharamsala Belt»
    parliamentary: rs-271-0274 / 271
    containment 0.014 · jaccard 0.004 · best containment 0.014 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-20d))
- 29536-00-hin [hin] «PARLIAMENT QUESTION: Earthquake Risk in Kangra-Chamba-Dharamsala Belt»
    parliamentary: rs-271-0274 / 271
    containment 0.014 · jaccard 0.004 · best containment 0.014 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-20d)) (record verdict via sibling document)
- 29545-00-eng [eng] «PARLIAMENT QUESTION: IMPLEMENTATION OF MISSION MAUSAM»
    parliamentary: rs-271-0529 / 271
    containment 0.475 · jaccard 0.396 · best containment 0.475 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ1d))
- 29545-00-hin [hin] «PARLIAMENT QUESTION: IMPLEMENTATION OF MISSION MAUSAM»
    parliamentary: rs-271-0529 / 271
    containment 0.475 · jaccard 0.396 · best containment 0.475 < 0.5 (best candidate rs-271-0529 (session 271, date 2026-07-23, Δ1d)) (record verdict via sibling document)
- 29555-00-eng [eng] «PARLIAMENT QUESTION: HEATWAVE FORECASTING AND EARLY WARNING SYSTEMS AT THE DISTR»
    parliamentary: rs-271-2129 / 271
    containment 0.035 · jaccard 0.022 · best containment 0.035 < 0.5 (best candidate rs-271-2129 (session 271, date 2026-08-06, Δ-13d))
- 29555-00-hin [hin] «PARLIAMENT QUESTION: HEATWAVE FORECASTING AND EARLY WARNING SYSTEMS AT THE DISTR»
    parliamentary: rs-271-2129 / 271
    containment 0.035 · jaccard 0.022 · best containment 0.035 < 0.5 (best candidate rs-271-2129 (session 271, date 2026-08-06, Δ-13d)) (record verdict via sibling document)
- 29570-00-eng [eng] «PARLIAMENT QUESTION: RAINFALL DEFICIT»
    parliamentary: rs-271-2914 / 271
    containment 0.037 · jaccard 0.011 · best containment 0.037 < 0.5 (best candidate rs-271-2914 (session 271, date 2026-08-13, Δ-20d))
- 29570-00-hin [hin] «PARLIAMENT QUESTION: RAINFALL DEFICIT»
    parliamentary: rs-271-2914 / 271
    containment 0.037 · jaccard 0.011 · best containment 0.037 < 0.5 (best candidate rs-271-2914 (session 271, date 2026-08-13, Δ-20d)) (record verdict via sibling document)
- 29578-00-eng [eng] «PARLIAMENT QUESTION: BHARAT FORECAST SYSTEM»
    parliamentary: rs-271-0274 / 271
    containment 0.012 · jaccard 0.003 · best containment 0.012 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-20d))
- 29578-00-hin [hin] «PARLIAMENT QUESTION: BHARAT FORECAST SYSTEM»
    parliamentary: rs-271-0274 / 271
    containment 0.012 · jaccard 0.003 · best containment 0.012 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ-20d)) (record verdict via sibling document)
- 29674-00-eng [eng] «PARLIAMENT QUESTION: LONG RANGE FORECAST SYSTEM»
    parliamentary: rs-271-2914 / 271
    containment 0.445 · jaccard 0.249 · best containment 0.445 < 0.5 (best candidate rs-271-2914 (session 271, date 2026-08-13, Δ-14d))
- 29674-00-hin [hin] «PARLIAMENT QUESTION: LONG RANGE FORECAST SYSTEM»
    parliamentary: rs-271-2914 / 271
    containment 0.445 · jaccard 0.249 · best containment 0.445 < 0.5 (best candidate rs-271-2914 (session 271, date 2026-08-13, Δ-14d)) (record verdict via sibling document)
- 29695-00-eng [eng] «PARLIAMENT QUESTION: NCCR Studies on Microplastic Pollution»
    parliamentary: rs-271-2132 / 271
    containment 0.020 · jaccard 0.007 · best containment 0.020 < 0.5 (best candidate rs-271-2132 (session 271, date 2026-08-06, Δ-7d))
- 29695-00-hin [hin] «PARLIAMENT QUESTION: NCCR Studies on Microplastic Pollution»
    parliamentary: rs-271-2132 / 271
    containment 0.020 · jaccard 0.007 · best containment 0.020 < 0.5 (best candidate rs-271-2132 (session 271, date 2026-08-06, Δ-7d)) (record verdict via sibling document)
- 29703-00-eng [eng] «PARLIAMENT QUESTION: Research Projects in Jammu and Kashmir»
    parliamentary: rs-271-2917 / 271
    containment 0.026 · jaccard 0.009 · best containment 0.026 < 0.5 (best candidate rs-271-2917 (session 271, date 2026-08-13, Δ-14d))
- 29703-00-hin [hin] «PARLIAMENT QUESTION: Research Projects in Jammu and Kashmir»
    parliamentary: rs-271-2917 / 271
    containment 0.026 · jaccard 0.009 · best containment 0.026 < 0.5 (best candidate rs-271-2917 (session 271, date 2026-08-13, Δ-14d)) (record verdict via sibling document)
- 29801-00-eng [eng] «PARLIAMENT QUESTION: Preparedness for EI Nino Conditions»
    parliamentary: rs-271-1323 / 271
    containment 0.028 · jaccard 0.016 · best containment 0.028 < 0.5 (best candidate rs-271-1323 (session 271, date 2026-07-30, Δ7d))
- 29801-00-hin [hin] «PARLIAMENT QUESTION: Preparedness for EI Nino Conditions»
    parliamentary: rs-271-1323 / 271
    containment 0.028 · jaccard 0.016 · best containment 0.028 < 0.5 (best candidate rs-271-1323 (session 271, date 2026-07-30, Δ7d)) (record verdict via sibling document)
- 29813-00-eng [eng] «PARLIAMENT QUESTION: ADVANCING MONSOON SCIENCE, CLIMATE RESEARCH AND WEATHER FOR»
    parliamentary: rs-271-0530 / 271
    containment 0.150 · jaccard 0.069 · best containment 0.150 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ14d))
- 29813-00-hin [hin] «PARLIAMENT QUESTION: ADVANCING MONSOON SCIENCE, CLIMATE RESEARCH AND WEATHER FOR»
    parliamentary: rs-271-0530 / 271
    containment 0.150 · jaccard 0.069 · best containment 0.150 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ14d)) (record verdict via sibling document)
- 29817-00-eng [eng] «PARLIAMENT QUESTION: WEATHER-BASED AGRO-ADVISOR SERVICES»
    parliamentary: rs-271-1318 / 271
    containment 0.295 · jaccard 0.176 · best containment 0.295 < 0.5 (best candidate rs-271-1318 (session 271, date 2026-07-30, Δ7d))
- 29817-00-hin [hin] «PARLIAMENT QUESTION: WEATHER-BASED AGRO-ADVISOR SERVICES»
    parliamentary: rs-271-1318 / 271
    containment 0.295 · jaccard 0.176 · best containment 0.295 < 0.5 (best candidate rs-271-1318 (session 271, date 2026-07-30, Δ7d)) (record verdict via sibling document)
- 29929-00-eng [eng] «PARLIAMENT QUESTION: FLOOD FORECASTING AND EARLY WARNING SYSTEM IN NORTHEAST IND»
    parliamentary: rs-271-0274 / 271
    containment 0.011 · jaccard 0.006 · best containment 0.011 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ0d))
- 29929-00-hin [hin] «PARLIAMENT QUESTION: FLOOD FORECASTING AND EARLY WARNING SYSTEM IN NORTHEAST IND»
    parliamentary: rs-271-0274 / 271
    containment 0.011 · jaccard 0.006 · best containment 0.011 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ0d)) (record verdict via sibling document)
- 29940-00-eng [eng] «PARLIAMENT QUESTION: Status of Early Warning Systems»
    parliamentary: rs-271-0274 / 271
    containment 0.027 · jaccard 0.010 · best containment 0.027 < 0.5 (best candidate rs-271-0274 (session 271, date 2026-08-13, Δ0d))
- 29945-00-eng [eng] «PARLIAMENT QUESTION: ADVANCED MARINE STATION FOR OCEAN BIOLOGY»
    parliamentary: rs-271-2917 / 271
    containment 0.023 · jaccard 0.012 · best containment 0.023 < 0.5 (best candidate rs-271-2917 (session 271, date 2026-08-13, Δ0d))
- 29945-00-hin [hin] «PARLIAMENT QUESTION: ADVANCED MARINE STATION FOR OCEAN BIOLOGY»
    parliamentary: rs-271-2917 / 271
    containment 0.023 · jaccard 0.012 · best containment 0.023 < 0.5 (best candidate rs-271-2917 (session 271, date 2026-08-13, Δ0d)) (record verdict via sibling document)
- 29953-00-eng [eng] «PARLIAMENT QUESTION: DEEP OCEAN MISSION»
    parliamentary: rs-271-1322 / 271
    containment 0.101 · jaccard 0.032 · best containment 0.101 < 0.5 (best candidate rs-271-1322 (session 271, date 2026-07-30, Δ14d))
- 29953-00-hin [hin] «PARLIAMENT QUESTION: DEEP OCEAN MISSION»
    parliamentary: rs-271-1322 / 271
    containment 0.101 · jaccard 0.032 · best containment 0.101 < 0.5 (best candidate rs-271-1322 (session 271, date 2026-07-30, Δ14d)) (record verdict via sibling document)
- 29961-00-eng [eng] «PARLIAMENT QUESTION: SOUTH-WEST MONSOON FORECAST»
    parliamentary: rs-271-1323 / 271
    containment 0.019 · jaccard 0.010 · best containment 0.019 < 0.5 (best candidate rs-271-1323 (session 271, date 2026-07-30, Δ14d))
- 29961-00-hin [hin] «PARLIAMENT QUESTION: SOUTH-WEST MONSOON FORECAST»
    parliamentary: rs-271-1323 / 271
    containment 0.019 · jaccard 0.010 · best containment 0.019 < 0.5 (best candidate rs-271-1323 (session 271, date 2026-07-30, Δ14d)) (record verdict via sibling document)
- 29969-00-eng [eng] «PARLIAMENT QUESTION: AI-Based Early Warning System»
    parliamentary: rs-271-0530 / 271
    containment 0.026 · jaccard 0.010 · best containment 0.026 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ21d))
- 29969-00-hin [hin] «PARLIAMENT QUESTION: AI-Based Early Warning System»
    parliamentary: rs-271-0530 / 271
    containment 0.026 · jaccard 0.010 · best containment 0.026 < 0.5 (best candidate rs-271-0530 (session 271, date 2026-07-23, Δ21d)) (record verdict via sibling document)

=== UNCOMPARABLE ===

(none)

DONE — read-only comparison; no crawler/corpus files modified.
