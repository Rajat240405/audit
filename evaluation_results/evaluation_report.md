# Parliamentary RAG - Retrieval Pipeline Evaluation Report
**Execution Date**: 2026-08-04 05:30:52 UTC
**Total Evaluation Queries**: 60 unique representative queries

## 1. Executive Summary & Overall Metrics
This report documents the scientific evaluation of the four retrieval pipeline iterations. The metric evaluations benchmark performance independently under identical query parameters.

| Metric | BM25 Only | Dense (FAISS) | Hybrid (RRF) | Hybrid + Cross-Encoder |
| :--- | :---: | :---: | :---: | :---: |
| Recall@1 (Hit Rate@1) | 0.0% | 0.0% | 0.0% | 0.0% |
| Recall@3 | 0.0% | 0.0% | 0.0% | 0.0% |
| Recall@5 (Hit Rate@5) | 0.0% | 0.0% | 0.0% | 0.0% |
| Recall@10 | 0.0% | 0.0% | 0.0% | 0.0% |
| Mean Reciprocal Rank (MRR) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| nDCG@5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| nDCG@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Average Rank | 100.00 | 100.00 | 100.00 | 100.00 |
| Mean Latency (ms) | 0.21 ms | 11.71 ms | 11.44 ms | 2545.94 ms |
| 95th Percentile Latency (ms) | 0.32 ms | 17.25 ms | 15.81 ms | 2831.86 ms |

## 2. Topic Category Breakdown (Recall@5)
Performance breakdown by topic domain category across each retrieval system.

| Topic Category | BM25 Only | Dense (FAISS) | Hybrid (RRF) | Hybrid + Cross-Encoder |
| :--- | :---: | :---: | :---: | :---: |
| COMMERCE AND INDUSTRY | 0.0% | 0.0% | 0.0% | 0.0% |
| RURAL DEVELOPMENT | 0.0% | 0.0% | 0.0% | 0.0% |
| AGRICULTURE AND FARMERS WELFARE | 0.0% | 0.0% | 0.0% | 0.0% |
| ELECTRONICS AND INFORMATION TECHNOLOGY | 0.0% | 0.0% | 0.0% | 0.0% |
| CONSUMER AFFAIRS, FOOD AND PUBLIC DISTRIBUTION | 0.0% | 0.0% | 0.0% | 0.0% |
| CIVIL AVIATION | 0.0% | 0.0% | 0.0% | 0.0% |
| TOURISM | 0.0% | 0.0% | 0.0% | 0.0% |
| JAL SHAKTI | 0.0% | 0.0% | 0.0% | 0.0% |
| FISHERIES, ANIMAL HUSBANDRY AND DAIRYING | 0.0% | 0.0% | 0.0% | 0.0% |
| HEALTH AND FAMILY WELFARE | 0.0% | 0.0% | 0.0% | 0.0% |
| MICRO, SMALL AND MEDIUM ENTERPRISES | 0.0% | 0.0% | 0.0% | 0.0% |
| FINANCE | 0.0% | 0.0% | 0.0% | 0.0% |
| YOUTH AFFAIRS AND SPORTS | 0.0% | 0.0% | 0.0% | 0.0% |
| EDUCATION | 0.0% | 0.0% | 0.0% | 0.0% |
| INFORMATION AND BROADCASTING | 0.0% | 0.0% | 0.0% | 0.0% |
| PETROLEUM AND NATURAL GAS | 0.0% | 0.0% | 0.0% | 0.0% |
| COMMUNICATION | 0.0% | 0.0% | 0.0% | 0.0% |
| SKILL DEVELOPMENT AND ENTREPRENEURSHIP | 0.0% | 0.0% | 0.0% | 0.0% |
| ENVIRONMENT, FOREST AND CLIMATE CHANGE | 0.0% | 0.0% | 0.0% | 0.0% |
| ROAD TRANSPORT AND HIGHWAYS | 0.0% | 0.0% | 0.0% | 0.0% |
| EXTERNAL AFFAIRS | 0.0% | 0.0% | 0.0% | 0.0% |
| PORTS, SHIPPING AND WATERWAYS | 0.0% | 0.0% | 0.0% | 0.0% |
| HOUSING AND URBAN AFFAIRS | 0.0% | 0.0% | 0.0% | 0.0% |
| LABOUR AND EMPLOYMENT | 0.0% | 0.0% | 0.0% | 0.0% |
| CORPORATE AFFAIRS | 0.0% | 0.0% | 0.0% | 0.0% |
| RAILWAYS | 0.0% | 0.0% | 0.0% | 0.0% |
| SOCIAL JUSTICE AND EMPOWERMENT | 0.0% | 0.0% | 0.0% | 0.0% |
| COOPERATION | 0.0% | 0.0% | 0.0% | 0.0% |
| TRIBAL AFFAIRS | 0.0% | 0.0% | 0.0% | 0.0% |
| FOOD PROCESSING INDUSTRIES | 0.0% | 0.0% | 0.0% | 0.0% |
| HOME AFFAIRS | 0.0% | 0.0% | 0.0% | 0.0% |

## 3. Visualizations
The following comparative charts were automatically generated and saved to the `evaluation_results/` folder:
* **Recall Comparison**: `recall_comparison.png`
* **MRR & nDCG Metrics**: `mrr_ndcg_comparison.png`
* **Latency Profile**: `latency_comparison.png`

## 4. Failure Analysis
Of the 60 queries, **60** failed to retrieve their expected targets in the top-5 final ranks under the complete Hybrid + Cross-Encoder pipeline.

| Failed Query | Expected ID | Top 1 Retrieved | Failure Stage | Possible Cause |
| :--- | :---: | :---: | :---: | :--- |
| trade infrastructure for export scheme | `18-6-1525` | `18-6-1392` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of assistance to odisha under mgnregs? | `18-6-0138` | `18-5-4162` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning request to business community | `18-2-2339` | `18-6-2750` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| production of millets | `18-5-0335` | `18-6-0002` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of dpdp rules? | `18-6-0550` | `18-6-2750` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning foodgrains to ration card holders | `18-2-1534` | `18-6-2750` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| digi yatra app | `18-2-0146` | `18-6-1392` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of promotion of religious places of rajasthan? | `18-4-1777` | `18-6-1392` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning irrigation projects under pmksy in tamil nadu | `18-4-4463` | `18-5-4162` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| primary fishermen cooperative society | `18-5-2570` | `18-5-4162` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of compliance of nhm guidelines? | `18-4-0314` | `18-6-1392` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning revival of closed msmes | `18-3-0689` | `18-6-2750` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| regulations for retail investors | `18-6-2473` | `18-6-0002` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of human resource development in sports? | `18-5-3612` | `18-6-1392` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning smart classrooms in madhya pradesh | `18-3-3313` | `18-4-4675` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| upgradation of community radio stations and doordarshan kendras | `18-5-4457` | `18-4-4675` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of analogy of posts link available on uidai website? | `18-5-4372` | `18-6-1392` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning export promotion mission | `18-5-0853` | `18-6-2750` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| city gas distribution in bengaluru | `18-2-2839` | `18-4-3433` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of projects approved under midh in andhra pradesh? | `18-6-0428` | `18-6-1392` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning pm-wani scheme | `18-4-4157` | `18-5-1727` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| scheme for skill development for youth of chhattisgarh | `18-3-0019` | `18-5-1179` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of healthy childhood through pure water and sanitation? | `18-5-2991` | `18-5-1872` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning forest land on lease | `18-2-2220` | `18-6-0002` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| construction of expressway from delhi to kmp | `18-5-0891` | `18-5-4189` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of prices of essential food items? | `18-5-1725` | `18-6-1392` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning enhancement of consular services | `18-3-3115` | `18-6-2750` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| cargo movement on national waterways | `18-6-2199` | `18-3-0904` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of housing projects? | `18-4-0681` | `18-6-0002` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning schemes launched for small and marginal farmers | `18-5-2579` | `18-3-3540` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| welfare schemes for rural and farm labourers | `18-4-0979` | `18-5-1526` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of internship to youths under pmis? | `18-4-0198` | `18-5-4162` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning proposed ecological sensitive areas  in goa | `18-4-1035` | `18-3-0904` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| railway projects in andhra pradesh | `18-4-0447` | `18-3-0416` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of guidelines for misleading advertisements? | `18-3-3841` | `18-6-1392` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning insurance for pwd | `18-4-0297` | `18-6-2750` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| financial and functional sustainability of integrated command centres in smart cities | `18-6-1848` | `18-6-1392` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of dairy sahakar yojana? | `18-6-2709` | `18-5-4162` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning train derailments and accidents in the country | `18-2-2747` | `18-5-1727` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| status of cuddalore port | `18-3-0894` | `18-3-0904` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of yuva sahakar scheme? | `18-4-0412` | `18-4-4675` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning ship repair facility in assam | `18-3-1982` | `18-3-0904` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| ambitious programme for tribal community | `18-3-0685` | `18-4-3628` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of quality of arable soil? | `18-6-1502` | `18-6-0002` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning women entrepreneurship schemes | `18-4-0929` | `18-5-1179` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| new railway lines for karnataka | `18-2-0401` | `18-3-0416` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of installation of train anti-collision systems? | `18-4-5156` | `18-3-0416` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning training of maldivian civil servants | `18-3-4153` | `18-6-2750` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| development of petrochemical sector | `18-5-3117` | `18-6-0002` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of pradhan mantri kisan sampada yojana? | `18-3-0500` | `18-5-4162` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning ai integrated early warning systems | `18-6-0332` | `18-6-2750` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| internet shutdowns | `18-4-2195` | `18-4-3628` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of development of new greenfield ports? | `18-3-0280` | `18-3-0904` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning guidelines for teachers during weather emergencies | `18-6-2376` | `18-6-2750` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| funds for apeda | `18-6-0234` | `18-4-3628` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of installation of cng pumps in bihar? | `18-4-0642` | `18-3-0416` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning safety audits of school and public facilities buildings | `18-5-4011` | `18-6-2750` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| national youth policy to empower the youth of nashik | `18-4-3909` | `18-4-4675` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| what are the details and active status of bharatpol portal? | `18-4-0252` | `18-3-0904` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |
| policy initiatives and development frameworks concerning air connectivity in hingoli, maharashtra | `18-6-0920` | `18-3-0904` | retrieval_retrieved_none | Expected document was not retrieved in first-stage (Dense & BM25 both missed). |