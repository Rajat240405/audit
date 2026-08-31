# MoES ↔ Parliamentary Q&A dedup exclusion report

Generated at ingestion. Only **confirmed** duplicates are excluded: EXACT_SHA and TEXTUALLY_NEAR_IDENTICAL. All other classes are preserved.

- MoES corpus root      : E:\audit2\data\.moes-website
- Parliamentary root   : E:\audit2\data\parliamentary-qa\rajya-sabha
- near_identical_threshold : 0.9
- related_threshold        : 0.5
- MoES PQ documents compared : 188
- EXCLUDED (confirmed dup)  : 20
  - EXACT_SHA: 0, TEXTUALLY_NEAR_IDENTICAL: 20, POTENTIALLY_CORRESPONDING: 41, POTENTIALLY_UNIQUE: 127, UNCOMPARABLE: 0

## Excluded documents

- `press-release/parliament-question-advancements-in-weather-forecasting/documents/01-27419-eng.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.946
    record moes-web-27419; parliamentary: rs-270-0041 / 270
    reason: containment 0.946 >= 0.9 vs rs-270-0041 (session 270, date 2026-01-29, Δ1d); title ratio 0.22; shingles 5-gram
- `press-release/parliament-question-advancements-in-weather-forecasting/documents/01-27419-hin.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.946
    record moes-web-27419; parliamentary: rs-270-0041 / 270
    reason: containment 0.946 >= 0.9 vs rs-270-0041 (session 270, date 2026-01-29, Δ1d); title ratio 0.22; shingles 5-gram (record verdict via sibling document)
- `press-release/parliament-question-improving-climate-and-weather-services/documents/01-27227-eng.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.954
    record moes-web-27227; parliamentary: rs-269-2113 / 269
    reason: containment 0.954 >= 0.9 vs rs-269-2113 (session 269, date 2025-12-18, Δ1d); title ratio 0.00; shingles 5-gram
- `press-release/parliament-question-improving-climate-and-weather-services/documents/01-27227-hin.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.954
    record moes-web-27227; parliamentary: rs-269-2113 / 269
    reason: containment 0.954 >= 0.9 vs rs-269-2113 (session 269, date 2025-12-18, Δ1d); title ratio 0.00; shingles 5-gram (record verdict via sibling document)
- `press-release/parliament-question-national-climate-services/documents/01-27166-eng.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.955
    record moes-web-27166; parliamentary: rs-269-1307 / 269
    reason: containment 0.955 >= 0.9 vs rs-269-1307 (session 269, date 2025-12-11, Δ1d); title ratio 0.00; shingles 5-gram
- `press-release/parliament-question-national-climate-services/documents/01-27166-hin.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.955
    record moes-web-27166; parliamentary: rs-269-1307 / 269
    reason: containment 0.955 >= 0.9 vs rs-269-1307 (session 269, date 2025-12-11, Δ1d); title ratio 0.00; shingles 5-gram (record verdict via sibling document)
- `press-release/parliament-question-ocean-observation-network-and-forecasting/documents/01-29761-eng.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.913
    record moes-web-29761; parliamentary: rs-271-1324 / 271
    reason: containment 0.913 >= 0.9 vs rs-271-1324 (session 271, date 2026-07-30, Δ1d); title ratio 0.09; shingles 5-gram
- `press-release/parliament-question-ocean-observation-network-and-forecasting/documents/01-29761-hin.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.913
    record moes-web-29761; parliamentary: rs-271-1324 / 271
    reason: containment 0.913 >= 0.9 vs rs-271-1324 (session 271, date 2026-07-30, Δ1d); title ratio 0.09; shingles 5-gram (record verdict via sibling document)
- `press-release/parliament-question-operational-forecasting-and-monitoring-mechanisms-2/documents/01-27061-eng.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.949
    record moes-web-27061; parliamentary: rs-269-0515 / 269
    reason: containment 0.949 >= 0.9 vs rs-269-0515 (session 269, date 2025-12-04, Δ1d); title ratio 0.00; shingles 5-gram
- `press-release/parliament-question-operational-forecasting-and-monitoring-mechanisms-2/documents/01-27061-hin.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.949
    record moes-web-27061; parliamentary: rs-269-0515 / 269
    reason: containment 0.949 >= 0.9 vs rs-269-0515 (session 269, date 2025-12-04, Δ1d); title ratio 0.00; shingles 5-gram (record verdict via sibling document)
- `press-release/parliament-question-operational-forecasting-and-monitoring-mechanisms/documents/01-26817-eng.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.946
    record moes-web-26817; parliamentary: rs-269-0515 / 269
    reason: containment 0.946 >= 0.9 vs rs-269-0515 (session 269, date 2025-12-04, Δ0d); title ratio 0.00; shingles 5-gram
- `press-release/parliament-question-operational-forecasting-and-monitoring-mechanisms/documents/01-26817-hin.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.946
    record moes-web-26817; parliamentary: rs-269-0515 / 269
    reason: containment 0.946 >= 0.9 vs rs-269-0515 (session 269, date 2025-12-04, Δ0d); title ratio 0.00; shingles 5-gram (record verdict via sibling document)
- `press-release/parliament-question-strengthening-climate-resilience-and-forecasting-capabilities/documents/01-29990-eng.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.916
    record moes-web-29990; parliamentary: rs-271-2915 / 271
    reason: containment 0.916 >= 0.9 vs rs-271-2915 (session 271, date 2026-08-13, Δ1d); title ratio 0.01; shingles 5-gram
- `press-release/parliament-question-strengthening-climate-resilience-and-forecasting-capabilities/documents/01-29990-hin.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.916
    record moes-web-29990; parliamentary: rs-271-2915 / 271
    reason: containment 0.916 >= 0.9 vs rs-271-2915 (session 271, date 2026-08-13, Δ1d); title ratio 0.01; shingles 5-gram (record verdict via sibling document)
- `press-release/parliament-question-strengthening-long-range-forecasting-capabilities/documents/01-29978-eng.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.912
    record moes-web-29978; parliamentary: rs-271-2914 / 271
    reason: containment 0.912 >= 0.9 vs rs-271-2914 (session 271, date 2026-08-13, Δ1d); title ratio 0.00; shingles 5-gram
- `press-release/parliament-question-strengthening-long-range-forecasting-capabilities/documents/01-29978-hin.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.912
    record moes-web-29978; parliamentary: rs-271-2914 / 271
    reason: containment 0.912 >= 0.9 vs rs-271-2914 (session 271, date 2026-08-13, Δ1d); title ratio 0.00; shingles 5-gram (record verdict via sibling document)
- `press-release/parliament-question-unpredictable-weather-patterns/documents/01-27172-eng.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.965
    record moes-web-27172; parliamentary: rs-269-1306 / 269
    reason: containment 0.965 >= 0.9 vs rs-269-1306 (session 269, date 2025-12-11, Δ1d); title ratio 0.00; shingles 5-gram
- `press-release/parliament-question-unpredictable-weather-patterns/documents/01-27172-hin.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.965
    record moes-web-27172; parliamentary: rs-269-1306 / 269
    reason: containment 0.965 >= 0.9 vs rs-269-1306 (session 269, date 2025-12-11, Δ1d); title ratio 0.00; shingles 5-gram (record verdict via sibling document)
- `press-release/parliament-question-weather-forecasts-by-regional-meteorological-centre-rmc-kolkata/documents/01-27073-eng.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.924
    record moes-web-27073; parliamentary: rs-269-0514 / 269
    reason: containment 0.924 >= 0.9 vs rs-269-0514 (session 269, date 2025-12-04, Δ1d); title ratio 0.00; shingles 5-gram
- `press-release/parliament-question-weather-forecasts-by-regional-meteorological-centre-rmc-kolkata/documents/01-27073-hin.pdf` [TEXTUALLY_NEAR_IDENTICAL] containment=0.924
    record moes-web-27073; parliamentary: rs-269-0514 / 269
    reason: containment 0.924 >= 0.9 vs rs-269-0514 (session 269, date 2025-12-04, Δ1d); title ratio 0.00; shingles 5-gram (record verdict via sibling document)

## Preserved documents

- `press-release/parliament-question-accuracy-of-forecast/documents/01-27234-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.004
- `press-release/parliament-question-accuracy-of-forecast/documents/01-27234-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.004
- `press-release/parliament-question-accuracy-of-weather-forecasting-systems/documents/01-27886-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.888
- `press-release/parliament-question-accuracy-of-weather-forecasts/documents/01-27724-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.012
- `press-release/parliament-question-accuracy-of-weather-forecasts/documents/01-27724-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.012
- `press-release/parliament-question-accurate-weather-forecasting-2/documents/01-28144-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.007
- `press-release/parliament-question-accurate-weather-forecasting-2/documents/01-28144-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.007
- `press-release/parliament-question-accurate-weather-forecasting/documents/01-27698-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.014
- `press-release/parliament-question-accurate-weather-forecasting/documents/01-27698-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.014
- `press-release/parliament-question-advanced-computer-simulation-models-to-improve-localised-weather-forecasting/documents/01-27130-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.045
- `press-release/parliament-question-advanced-computer-simulation-models-to-improve-localised-weather-forecasting/documents/01-27130-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.045
- `press-release/parliament-question-advanced-marine-station-for-ocean-biology/documents/01-29945-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.023
- `press-release/parliament-question-advanced-marine-station-for-ocean-biology/documents/01-29945-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.023
- `press-release/parliament-question-advanced-meteorological-infrastructure/documents/01-28339-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.005
- `press-release/parliament-question-advanced-meteorological-infrastructure/documents/01-28339-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.005
- `press-release/parliament-question-advancing-monsoon-science-climate-research-and-weather-forecasting/documents/01-29813-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.026
- `press-release/parliament-question-advancing-monsoon-science-climate-research-and-weather-forecasting/documents/01-29813-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.026
- `press-release/parliament-question-adverse-effects-of-climate-change-on-himalayas/documents/01-26811-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.010
- `press-release/parliament-question-adverse-effects-of-climate-change-on-himalayas/documents/01-26811-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.010
- `press-release/parliament-question-ai-and-geospatial-technologies-for-weather-forecasting/documents/01-29995-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.018
- `press-release/parliament-question-ai-and-geospatial-technologies-for-weather-forecasting/documents/01-29995-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.018
- `press-release/parliament-question-ai-based-early-warning-system/documents/01-29969-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.011
- `press-release/parliament-question-ai-based-early-warning-system/documents/01-29969-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.011
- `press-release/parliament-question-ai-in-weather-forecasting-2/documents/01-28159-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.026
- `press-release/parliament-question-ai-in-weather-forecasting-2/documents/01-28159-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.026
- `press-release/parliament-question-automatic-weather-stations/documents/01-28247-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.003
- `press-release/parliament-question-automatic-weather-stations/documents/01-28247-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.003
- `press-release/parliament-question-bharat-forecast-system-4/documents/01-29578-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.785
- `press-release/parliament-question-bharat-forecast-system-4/documents/01-29578-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.785
- `press-release/parliament-question-brahmaputra-flood-forecasting-and-climate-resilience-mission/documents/01-29735-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.851
- `press-release/parliament-question-brahmaputra-flood-forecasting-and-climate-resilience-mission/documents/01-29735-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.851
- `press-release/parliament-question-coastline-of-the-country/documents/01-27099-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.000
- `press-release/parliament-question-coastline-of-the-country/documents/01-27099-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.000
- `press-release/parliament-question-deep-ocean-mission-2/documents/01-27154-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.013
- `press-release/parliament-question-deep-ocean-mission-2/documents/01-27154-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.013
- `press-release/parliament-question-deep-ocean-mission-3/documents/01-29953-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.032
- `press-release/parliament-question-deep-ocean-mission-3/documents/01-29953-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.032
- `press-release/parliament-question-deep-sea-exploration-contracts/documents/01-27086-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.801
- `press-release/parliament-question-deep-sea-exploration-contracts/documents/01-27086-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.801
- `press-release/parliament-question-deep-sea-exploration-for-minerals-oil-and-natural-gas/documents/01-27094-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.869
- `press-release/parliament-question-deep-sea-exploration-for-minerals-oil-and-natural-gas/documents/01-27094-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.869
- `press-release/parliament-question-deep-sea-mining/documents/01-28172-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.014
- `press-release/parliament-question-deep-sea-mining/documents/01-28172-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.014
- `press-release/parliament-question-doppler-radar-coverage-in-lahaul-and-spiti-and-kinnaur/documents/01-29752-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.754
- `press-release/parliament-question-doppler-radar-coverage-in-lahaul-and-spiti-and-kinnaur/documents/01-29752-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.754
- `press-release/parliament-question-doppler-radar-station-at-balasore-2/documents/01-24224-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.678
- `press-release/parliament-question-early-warning-systems-for-floods-and-cyclones/documents/01-28117-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.010
- `press-release/parliament-question-early-warning-systems-for-floods-and-cyclones/documents/01-28117-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.010
- `press-release/parliament-question-earthquake-risk-in-kangra-chamba-dharamsala-belt/documents/01-29536-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.029
- `press-release/parliament-question-earthquake-risk-in-kangra-chamba-dharamsala-belt/documents/01-29536-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.029
- `press-release/parliament-question-earthquake-vulnerability-and-steps-taken-to-mitigate-the-risk/documents/01-28355-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.023
- `press-release/parliament-question-earthquake-vulnerability-and-steps-taken-to-mitigate-the-risk/documents/01-28355-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.023
- `press-release/parliament-question-el-nino-effect-on-monsoon-and-rainfall-2/documents/01-24227-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.118
- `press-release/parliament-question-expansion-of-ocean-observation-network/documents/01-29688-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.521
- `press-release/parliament-question-expansion-of-ocean-observation-network/documents/01-29688-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.521
- `press-release/parliament-question-extreme-weather-events/documents/01-28184-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.255
- `press-release/parliament-question-extreme-weather-events/documents/01-28184-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.255
- `press-release/parliament-question-financial-lapses-in-institutions-under-the-ministry-of-earth-sciences/documents/01-28346-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.019
- `press-release/parliament-question-financial-lapses-in-institutions-under-the-ministry-of-earth-sciences/documents/01-28346-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.019
- `press-release/parliament-question-flood-forecasting-and-early-warning-system-in-northeast-india/documents/01-29929-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.004
- `press-release/parliament-question-flood-forecasting-and-early-warning-system-in-northeast-india/documents/01-29929-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.004
- `press-release/parliament-question-forecast-system/documents/01-27437-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.003
- `press-release/parliament-question-forecast-system/documents/01-27437-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.003
- `press-release/parliament-question-forecasting-of-heavy-rains-and-landslides/documents/01-27207-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.038
- `press-release/parliament-question-forecasting-of-heavy-rains-and-landslides/documents/01-27207-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.038
- `press-release/parliament-question-fourth-global-coral-bleaching-event-2/documents/01-24240-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.002
- `press-release/parliament-question-heatwave-forecasting-and-early-warning-systems-at-the-district-level/documents/01-29555-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.031
- `press-release/parliament-question-heatwave-forecasting-and-early-warning-systems-at-the-district-level/documents/01-29555-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.031
- `press-release/parliament-question-high-risk-seismic-categorisation-of-the-himalayan-region/documents/01-27188-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.010
- `press-release/parliament-question-high-risk-seismic-categorisation-of-the-himalayan-region/documents/01-27188-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.010
- `press-release/parliament-question-impact-of-el-nino-on-indian-monsoon/documents/01-29744-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.796
- `press-release/parliament-question-impact-of-el-nino-on-indian-monsoon/documents/01-29744-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.796
- `press-release/parliament-question-implementation-of-mission-mausam-2/documents/01-29545-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.446
- `press-release/parliament-question-implementation-of-mission-mausam-2/documents/01-29545-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.446
- `press-release/parliament-question-implementation-of-mission-mausam-3/documents/01-29598-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.842
- `press-release/parliament-question-implementation-of-mission-mausam-3/documents/01-29598-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.842
- `press-release/parliament-question-implementation-of-mission-mausam-in-the-north-east-region-2/documents/01-24210-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.100
- `press-release/parliament-question-implementation-of-mission-mausam/documents/01-28149-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.881
- `press-release/parliament-question-implementation-of-mission-mausam/documents/01-28149-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.881
- `press-release/parliament-question-improving-weather-forecasting-capabilities/documents/01-27879-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.023
- `press-release/parliament-question-improving-weather-forecasting-capabilities/documents/01-27879-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.023
- `press-release/parliament-question-indigenious-warning-systems/documents/01-30003-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.015
- `press-release/parliament-question-indigenious-warning-systems/documents/01-30003-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.015
- `press-release/parliament-question-installation-of-weather-stations/documents/01-27873-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.792
- `press-release/parliament-question-installation-of-weather-stations/documents/01-27873-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.792
- `press-release/parliament-question-long-range-forecast-system/documents/01-29674-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.445
- `press-release/parliament-question-long-range-forecast-system/documents/01-29674-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.445
- `press-release/parliament-question-maitri-2-station/documents/01-27145-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.005
- `press-release/parliament-question-maitri-2-station/documents/01-27145-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.005
- `press-release/parliament-question-mapping-of-heatwaves/documents/01-26826-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.015
- `press-release/parliament-question-mapping-of-heatwaves/documents/01-26826-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.015
- `press-release/parliament-question-marine-mining-and-weather-forecasting/documents/01-27080-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.882
- `press-release/parliament-question-marine-mining-and-weather-forecasting/documents/01-27080-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.882
- `press-release/parliament-question-mission-mausam-2/documents/01-24211-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.220
- `press-release/parliament-question-mission-mausam-5/documents/01-27238-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.706
- `press-release/parliament-question-mission-mausam-5/documents/01-27238-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.706
- `press-release/parliament-question-mission-mausam-6/documents/01-28125-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.706
- `press-release/parliament-question-mission-mausam-6/documents/01-28125-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.706
- `press-release/parliament-question-mission-mausam-7/documents/01-28239-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.025
- `press-release/parliament-question-mission-mausam-7/documents/01-28239-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.025
- `press-release/parliament-question-mission-mausam-8/documents/01-30011-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.117
- `press-release/parliament-question-mission-mausam-8/documents/01-30011-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.117
- `press-release/parliament-question-mission-mausam-in-odisha/documents/01-28100-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.008
- `press-release/parliament-question-mission-mausam-in-odisha/documents/01-28100-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.008
- `press-release/parliament-question-mission-mausam-to-boost-the-radar-network-2/documents/01-24234-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.046
- `press-release/parliament-question-monitoring-system-for-risk-prone-areas/documents/01-27219-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.856
- `press-release/parliament-question-monitoring-system-for-risk-prone-areas/documents/01-27219-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.856
- `press-release/parliament-question-monsoon-forecast-accuracy-and-improvements/documents/01-27917-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.806
- `press-release/parliament-question-monsoon-forecast-accuracy-and-improvements/documents/01-27917-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.806
- `press-release/parliament-question-monsoon-forecasting-and-climate-resilience-in-assam/documents/01-27714-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.010
- `press-release/parliament-question-monsoon-forecasting-and-climate-resilience-in-assam/documents/01-27714-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.010
- `press-release/parliament-question-monsoon-prediction-2/documents/01-24195-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.008
- `press-release/parliament-question-monsoon-warning-system/documents/01-27201-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.003
- `press-release/parliament-question-monsoon-warning-system/documents/01-27201-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.003
- `press-release/parliament-question-nccr-studies-on-microplastic-pollution/documents/01-29695-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.006
- `press-release/parliament-question-nccr-studies-on-microplastic-pollution/documents/01-29695-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.006
- `press-release/parliament-question-ocean-mining-and-technology-upgradation/documents/01-27195-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.430
- `press-release/parliament-question-ocean-mining-and-technology-upgradation/documents/01-27195-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.430
- `press-release/parliament-question-offshore-deep-sea-mining-in-kerala/documents/01-27136-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.004
- `press-release/parliament-question-offshore-deep-sea-mining-in-kerala/documents/01-27136-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.004
- `press-release/parliament-question-performance-of-early-warning-systems/documents/01-27860-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.012
- `press-release/parliament-question-performance-of-early-warning-systems/documents/01-27860-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.012
- `press-release/parliament-question-plan-to-improve-weather-forecasting-2/documents/01-24209-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.025
- `press-release/parliament-question-preparedness-for-ei-nino-conditions/documents/01-29801-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.028
- `press-release/parliament-question-preparedness-for-ei-nino-conditions/documents/01-29801-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.028
- `press-release/parliament-question-prithvi-vigyan-scheme/documents/01-27708-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.321
- `press-release/parliament-question-prithvi-vigyan-scheme/documents/01-27708-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.321
- `press-release/parliament-question-promotion-of-blue-economy-2/documents/01-24220-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.073
- `press-release/parliament-question-rainfall-deficit/documents/01-29570-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.019
- `press-release/parliament-question-rainfall-deficit/documents/01-29570-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.019
- `press-release/parliament-question-real-time-weather-updates/documents/01-28256-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.839
- `press-release/parliament-question-real-time-weather-updates/documents/01-28256-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.839
- `press-release/parliament-question-research-and-training-programmes/documents/01-28132-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.007
- `press-release/parliament-question-research-and-training-programmes/documents/01-28132-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.007
- `press-release/parliament-question-research-projects-in-jammu-and-kashmir/documents/01-29703-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.010
- `press-release/parliament-question-research-projects-in-jammu-and-kashmir/documents/01-29703-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.010
- `press-release/parliament-question-research-projects-under-prithvi-scheme/documents/01-27067-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.827
- `press-release/parliament-question-research-projects-under-prithvi-scheme/documents/01-27067-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.827
- `press-release/parliament-question-rising-threats-of-climate-change/documents/01-27212-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.053
- `press-release/parliament-question-rising-threats-of-climate-change/documents/01-27212-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.053
- `press-release/parliament-question-south-west-monsoon-forecast/documents/01-29961-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.005
- `press-release/parliament-question-south-west-monsoon-forecast/documents/01-29961-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.005
- `press-release/parliament-question-status-of-early-warning-systems/documents/01-29940-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.017
- `press-release/parliament-question-status-of-implementation-of-mission-mausam/documents/01-28194-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.891
- `press-release/parliament-question-status-of-implementation-of-mission-mausam/documents/01-28194-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.891
- `press-release/parliament-question-strengtheing-radar-infrastructure-in-kerala/documents/01-27893-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.849
- `press-release/parliament-question-strengtheing-radar-infrastructure-in-kerala/documents/01-27893-hin.pdf` [POTENTIALLY_CORRESPONDING] containment=0.849
- `press-release/parliament-question-studies-for-climate-change/documents/01-28265-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.048
- `press-release/parliament-question-studies-for-climate-change/documents/01-28265-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.048
- `press-release/parliament-question-studies-for-coastal-erosion/documents/01-27730-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.004
- `press-release/parliament-question-studies-for-coastal-erosion/documents/01-27730-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.004
- `press-release/parliament-question-studies-to-assess-the-impact-of-extreme-weather-conditions/documents/01-27689-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.029
- `press-release/parliament-question-studies-to-assess-the-impact-of-extreme-weather-conditions/documents/01-27689-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.029
- `press-release/parliament-question-upgradation-in-observation-network/documents/01-27428-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.013
- `press-release/parliament-question-upgradation-in-observation-network/documents/01-27428-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.013
- `press-release/parliament-question-viksit-bharat-2/documents/01-24229-eng.pdf` [POTENTIALLY_CORRESPONDING] containment=0.656
- `press-release/parliament-question-vulnerability-of-coastal-region-to-flood-and-sea-level-rise/documents/01-27151-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.009
- `press-release/parliament-question-vulnerability-of-coastal-region-to-flood-and-sea-level-rise/documents/01-27151-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.009
- `press-release/parliament-question-weather-and-climate-services/documents/01-27160-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.128
- `press-release/parliament-question-weather-and-climate-services/documents/01-27160-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.128
- `press-release/parliament-question-weather-based-agro-advisor-services/documents/01-29817-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.032
- `press-release/parliament-question-weather-based-agro-advisor-services/documents/01-29817-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.032
- `press-release/parliament-question-weather-forecasting-and-early-warnings/documents/01-29586-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.142
- `press-release/parliament-question-weather-forecasting-and-early-warnings/documents/01-29586-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.142
- `press-release/parliament-question-winter-forecast-by-imd/documents/01-26829-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.004
- `press-release/parliament-question-winter-forecast-by-imd/documents/01-26829-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.004
- `press-release/parliament-questions-major-improvements-in-weather-forecasting/documents/01-28112-eng.pdf` [POTENTIALLY_UNIQUE] containment=0.200
- `press-release/parliament-questions-major-improvements-in-weather-forecasting/documents/01-28112-hin.pdf` [POTENTIALLY_UNIQUE] containment=0.200

DONE — read-only comparison; no crawler/corpus files modified.
