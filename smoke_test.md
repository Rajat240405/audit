(.venv) PS E:\audit2> retrieve query "GST collection"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 7352.67it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: GST collection

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 5267.15it/s]
                                   Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry    ┃                  Score ┃ Question (excerpt)             ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-3-0981 │ FINANCE     │  5.1208 (dense: 0.662) │ GOVERNMENT OF INDIA            │
│     │           │             │                        │ MINISTRY OF FINANCE            │
│     │           │             │                        │ DEPARTMENT OF R...             │
├─────┼───────────┼─────────────┼────────────────────────┼────────────────────────────────┤
│ 2   │ 18-5-2373 │ FINANCE     │  1.2550 (dense: 0.547) │ GOVERNMENT OF INDIA            │
│     │           │             │                        │ MINISTRY OF FINANCE            │
│     │           │             │                        │ DEPARTMENT OF R...             │
├─────┼───────────┼─────────────┼────────────────────────┼────────────────────────────────┤
│ 3   │ 18-6-1451 │ COOPERATION │ -0.3431 (dense: 0.439) │ GOVERNMENT OF INDIA            │
│     │           │             │                        │ MINISTRY OF COOPERATION        │
│     │           │             │                        │                                │
│     │           │             │                        │ LOK...                         │
├─────┼───────────┼─────────────┼────────────────────────┼────────────────────────────────┤
│ 4   │ 18-4-0138 │ FINANCE     │ -0.4999 (dense: 0.487) │ GOVERNMENT OF INDIA            │
│     │           │             │                        │ MINISTRY OF FINANCE            │
│     │           │             │                        │ DEPARTMENT OF R...             │
├─────┼───────────┼─────────────┼────────────────────────┼────────────────────────────────┤
│ 5   │ 18-3-1128 │ FINANCE     │ -0.5825 (dense: 0.468) │ GOVERNMENT OF INDIA            │
│     │           │             │                        │ MINISTRY OF FINANCE            │
│     │           │             │                        │ DEPARTMENT OF R...             │
└─────┴───────────┴─────────────┴────────────────────────┴────────────────────────────────┘

Retrieval: 8188ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 16,309 chars | Estimated tokens: 4,077
✓ Prompt size is well within effective context window (4,077 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
╭─────────────────────────────────────────────────────── Generated Answer ───────────────────────────────────────────────────────╮
│ The provided context does not contain sufficient information to answer this question directly. The context discusses various   │
│ aspects of GST, including its impact on health insurance premiums and cooperatives, but does not provide specific details      │
│ about the increase in GST collection or any related data.                                                                      │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Model: qwen2.5:3b | Tokens: 4146 | Latency: 7672ms | Sources: 18-3-0981, 18-5-2373, 18-6-1451, 18-4-0138, 18-3-1128
(.venv) PS E:\audit2> retrieve query "Income tax exemption"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 6718.30it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: Income tax exemption

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 6296.67it/s]
                                                   Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry                            ┃                  Score ┃ Question (excerpt)                     ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-4-0143 │ CULTURE                             │ -1.5617 (dense: 0.240) │ GOVERNMENT OF INDIA                    │
│     │           │                                     │                        │ MINISTRY OF CULTURE                    │
│     │           │                                     │                        │                                        │
│     │           │                                     │                        │ LOK SABHA                              │
│     │           │                                     │                        │ UN...                                  │
├─────┼───────────┼─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┤
│ 2   │ 18-2-1294 │ AGRICULTURE AND FARMERS WELFARE     │ -2.1602 (dense: 0.381) │ O.I.H.                                 │
│     │           │                                     │                        │ GOVERNMENT OF INDIA                    │
│     │           │                                     │                        │ MINISTRY OF AGRICULTURE AND ...        │
├─────┼───────────┼─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┤
│ 3   │ 18-3-4043 │ MICRO, SMALL AND MEDIUM ENTERPRISES │ -3.8941 (dense: 0.434) │ GOVERNMENT OF INDIA                    │
│     │           │                                     │                        │                                        │
│     │           │                                     │                        │ MINISTRY OF MICRO, SMALL AND MEDIUM... │
├─────┼───────────┼─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┤
│ 4   │ 18-4-5583 │ HEALTH AND FAMILY WELFARE           │ -5.1357 (dense: 0.302) │ GOVERNMENT OF INDIA                    │
│     │           │                                     │                        │ MINISTRY OF HEALTH AND FAMILYWELFARE   │
│     │           │                                     │                        │ ...                                    │
├─────┼───────────┼─────────────────────────────────────┼────────────────────────┼────────────────────────────────────────┤
│ 5   │ 18-4-2775 │ COOPERATION                         │                -5.4178 │ GOVERNMENT OF INDIA                    │
│     │           │                                     │                        │ MINISTRY OF COOPERATION                │
│     │           │                                     │                        │                                        │
│     │           │                                     │                        │ LOK SABHA...                           │
└─────┴───────────┴─────────────────────────────────────┴────────────────────────┴────────────────────────────────────────┘

Retrieval: 8094ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 25,397 chars | Estimated tokens: 6,349
✓ Prompt size is well within effective context window (6,349 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
╭─────────────────────────────────────────────────────── Generated Answer ───────────────────────────────────────────────────────╮
│ The provided context does not contain sufficient information to answer the question about income tax exemptions for MSME       │
│ sector.                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Model: qwen2.5:3b | Tokens: 6374 | Latency: 98547ms | Sources: 18-4-0143, 18-2-1294, 18-3-4043, 18-4-5583, 18-4-2775
(.venv) PS E:\audit2> retrieve query "National Livestock Mission"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 6980.34it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: National Livestock Mission

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 6059.87it/s]
                                                      Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry                                 ┃                  Score ┃ Question (excerpt)                      ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-6-0430 │ FISHERIES, ANIMAL HUSBANDRY AND DAIRYING │  7.3681 (dense: 0.544) │ GOVERNMENT OF INDIA                     │
│     │           │                                          │                        │ MINISTRY OF FISHERIES, ANIMAL HUSBAN... │
├─────┼───────────┼──────────────────────────────────────────┼────────────────────────┼─────────────────────────────────────────┤
│ 2   │ 18-4-0418 │ FISHERIES, ANIMAL HUSBANDRY AND DAIRYING │  6.0898 (dense: 0.452) │ GOVERNMENT OF INDIA                     │
│     │           │                                          │                        │ MINISTRY OF FISHERIES, ANIMAL HUSBAN... │
├─────┼───────────┼──────────────────────────────────────────┼────────────────────────┼─────────────────────────────────────────┤
│ 3   │ 18-4-1255 │ FISHERIES, ANIMAL HUSBANDRY AND DAIRYING │  3.4374 (dense: 0.463) │ GOVERNMENT OF INDIA                     │
│     │           │                                          │                        │ MINISTRY OF FISHERIES, ANIMAL HUSBAN... │
├─────┼───────────┼──────────────────────────────────────────┼────────────────────────┼─────────────────────────────────────────┤
│ 4   │ 18-4-0409 │ PANCHAYATI RAJ                           │                 0.1797 │ 1                                       │
│     │           │                                          │                        │ GOVERNMENT OF INDIA                     │
│     │           │                                          │                        │ MINISTRY OF PANCHAYATIRAJ               │
│     │           │                                          │                        │ LOK SABHA...                            │
├─────┼───────────┼──────────────────────────────────────────┼────────────────────────┼─────────────────────────────────────────┤
│ 5   │ 18-4-2576 │ ENVIRONMENT, FOREST AND CLIMATE CHANGE   │ -2.4187 (dense: 0.344) │ GOVERNMENT OF INDIA                     │
│     │           │                                          │                        │ MINISTRY OF ENVIRONMENT, FOREST AND ... │
└─────┴───────────┴──────────────────────────────────────────┴────────────────────────┴─────────────────────────────────────────┘

Retrieval: 8640ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 29,291 chars | Estimated tokens: 7,322
✓ Prompt size is well within effective context window (7,322 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
╭─────────────────────────────────────────────────────── Generated Answer ───────────────────────────────────────────────────────╮
│ Certainly, here is a concise and coherent response for the National Livestock Mission (NLM) based on the information provided: │
│                                                                                                                                │
│ ---                                                                                                                            │
│                                                                                                                                │
│ **MINISTER OF STATE FOR PANCHAYATIRAJ**                                                                                        │
│                                                                                                                                │
│ Prof. S.P. Singh Baghel                                                                                                        │
│                                                                                                                                │
│ (a) The details regarding the number of forest area, nurseries, and ecosystem restoration projects undertaken under the        │
│ National Mission for Green India (NMGI), year-wise, state-wise including Andhra Pradesh are detailed in Annexure-I.            │
│                                                                                                                                │
│ (b) The details regarding the number of village and community institutions strengthened such as Gram Panchayats, Joint Forest  │
│ Management Committee under the Scheme, year-wise, state-wise including Andhra Pradesh are detailed in Annexure-II.             │
│                                                                                                                                │
│ (c) The total funds allocated, released, and utilized under the National Mission for Green India (NMGI), since its inception   │
│ year, state-wise including Andhra Pradesh are detailed in Annexure-III.                                                        │
│                                                                                                                                │
│ (d) The number of community foresters supported under the Mission, year-wise, state-wise including Andhra Pradesh are detailed │
│ in Annexure-IV.                                                                                                                │
│                                                                                                                                │
│ ---                                                                                                                            │
│                                                                                                                                │
│ **Annexures:**                                                                                                                 │
│                                                                                                                                │
│ 1. **Annexure-I:** State-wise details of number of intervention landscapes, JFMCs involved, villages covered, nurseries        │
│ established and area taken up for eco-restoration under the NMGI from 2015-16 till date.                                       │
│                                                                                                                                │
│ 2. **Annexure-II:** State-wise & Year-wise funds released from FY 2015-16 till date under NMGI (Rs. in Crores).                │
│                                                                                                                                │
│ 3. **Annexure-III:** Total funds allocated, released, and utilized under the National Mission for Green India (NMGI), since    │
│ its inception year, state-wise including Andhra Pradesh.                                                                       │
│                                                                                                                                │
│ 4. **Annexure-IV:** Number of community foresters supported under the Mission, year-wise, state-wise including Andhra Pradesh. │
│                                                                                                                                │
│ ---                                                                                                                            │
│                                                                                                                                │
│ This structure ensures that all parts of the question are addressed comprehensively and coherently within a single response.   │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Model: qwen2.5:3b | Tokens: 7766 | Latency: 79359ms | Sources: 18-6-0430, 18-4-0418, 18-4-1255, 18-4-0409, 18-4-2576
(.venv) PS E:\audit2> retrieve query "National Livestock Mission"
Traceback (most recent call last):
  File "<frozen runpy></frozen>", line 198, in _run_module_as_main
  File "<frozen runpy></frozen>", line 88, in _run_code
  File "E:\audit2\.venv\Scripts\retrieve.exe\__main__.py", line 4, in <module></module>
  File "E:\audit2\src\retrieval\__init__.py", line 2, in <module></module>
    from src.retrieval.hybrid.pipeline import HybridRAGPipeline
  File "E:\audit2\src\retrieval\hybrid\__init__.py", line 3, in <module></module>
    from src.retrieval.hybrid.embedder import Embedder
  File "E:\audit2\src\retrieval\hybrid\embedder.py", line 24, in <module></module>
    import torch
  File "E:\audit2\.venv\Lib\site-packages\torch\__init__.py", line 2229, in <module></module>
    from torch import _VF as _VF, functional as functional  # usort: skip
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\torch\functional.py", line 8, in <module></module>
    import torch.nn.functional as F
  File "E:\audit2\.venv\Lib\site-packages\torch\nn\__init__.py", line 8, in <module></module>
    from torch.nn.modules import *  # usort: skip # noqa: F403
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\torch\nn\modules\__init__.py", line 35, in <module></module>
    from .batchnorm import (
  File "E:\audit2\.venv\Lib\site-packages\torch\nn\modules\batchnorm.py", line 9, in <module></module>
    from ._functions import SyncBatchNorm as sync_batch_norm
  File "E:\audit2\.venv\Lib\site-packages\torch\nn\modules\_functions.py", line 4, in <module></module>
    from torch.autograd.function import Function
  File "E:\audit2\.venv\Lib\site-packages\torch\autograd\__init__.py", line 16, in <module></module>
    from torch import _vmap_internals
  File "E:\audit2\.venv\Lib\site-packages\torch\_vmap_internals.py", line 9, in <module></module>
    from torch.utils._pytree import _broadcast_to_and_flatten, tree_flatten, tree_unflatten
  File "E:\audit2\.venv\Lib\site-packages\torch\utils\_pytree.py", line 970, in <module></module>
    _private_register_pytree_node(
  File "E:\audit2\.venv\Lib\site-packages\torch\utils\_pytree.py", line 611, in _private_register_pytree_node
    from torch._library.opaque_object import is_opaque_type
  File "E:\audit2\.venv\Lib\site-packages\torch\_library\__init__.py", line 1, in <module></module>
    import torch._library.autograd
  File "E:\audit2\.venv\Lib\site-packages\torch\_library\autograd.py", line 7, in <module></module>
    from torch import _C, _ops, autograd, Tensor
  File "E:\audit2\.venv\Lib\site-packages\torch\_ops.py", line 18, in <module></module>
    from torch._functorch.pyfunctorch import dispatch_functorch, TransformType
  File "<frozen importlib._bootstrap></frozen>", line 1176, in _find_and_load
  File "<frozen importlib._bootstrap></frozen>", line 1138, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap></frozen>", line 1078, in _find_spec
  File "<frozen importlib._bootstrap_external></frozen>", line 1507, in find_spec
  File "<frozen importlib._bootstrap_external></frozen>", line 1479, in _get_spec
  File "<frozen importlib._bootstrap_external></frozen>", line 1634, in find_spec
KeyboardInterrupt
(.venv) PS E:\audit2> retrieve query "PM Kisan scheme"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 6678.11it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: PM Kisan scheme

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 5976.10it/s]
                                                      Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry                        ┃                 Score ┃ Question (excerpt)                                 ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-4-3963 │ AGRICULTURE AND FARMERS WELFARE │ 7.2202 (dense: 0.658) │ O.I.H.                                             │
│     │           │                                 │                       │ GOVERNMENT OF INDIA                                │
│     │           │                                 │                       │ MINISTRY OF AGRICULTURE AND ...                    │
├─────┼───────────┼─────────────────────────────────┼───────────────────────┼────────────────────────────────────────────────────┤
│ 2   │ 18-4-1991 │ AGRICULTURE AND FARMERS WELFARE │ 6.8508 (dense: 0.648) │ GOVERNMENT OF INDIA                                │
│     │           │                                 │                       │ MINISTRY OF AGRICULTURE AND FARMERS ...            │
├─────┼───────────┼─────────────────────────────────┼───────────────────────┼────────────────────────────────────────────────────┤
│ 3   │ 18-3-2388 │ AGRICULTURE AND FARMERS WELFARE │ 6.3172 (dense: 0.542) │ O.I.H.                                             │
│     │           │                                 │                       │ GOVERNMENT OF INDIA                                │
│     │           │                                 │                       │ MINISTRY OF AGRICULTURE AND ...                    │
├─────┼───────────┼─────────────────────────────────┼───────────────────────┼────────────────────────────────────────────────────┤
│ 4   │ 18-2-1294 │ AGRICULTURE AND FARMERS WELFARE │ 6.2628 (dense: 0.512) │ O.I.H.                                             │
│     │           │                                 │                       │ GOVERNMENT OF INDIA                                │
│     │           │                                 │                       │ MINISTRY OF AGRICULTURE AND ...                    │
├─────┼───────────┼─────────────────────────────────┼───────────────────────┼────────────────────────────────────────────────────┤
│ 5   │ 18-2-1678 │ HOUSING AND URBAN AFFAIRS       │ 5.3545 (dense: 0.507) │ O.I.H. GOVERNMENT OF INDIA MINISTRY OF HOUSING AND │
│     │           │                                 │                       │ URBAN ...                                          │
└─────┴───────────┴─────────────────────────────────┴───────────────────────┴────────────────────────────────────────────────────┘

Retrieval: 12531ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 20,570 chars | Estimated tokens: 5,142
✓ Prompt size is well within effective context window (5,142 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
╭─────────────────────────────────────────────────────── Generated Answer ───────────────────────────────────────────────────────╮
│ The PM-KISAN scheme is a central sector scheme launched in February 2019 by the Hon'ble Prime Minister to supplement the       │
│ financial needs of cultivable land-holding farmers. Under this scheme, a financial benefit of Rs 6,000/- per year is           │
│ transferred in three equal instalments into the Aadhaar seeded bank accounts of farmers through Direct Benefit Transfer (DBT)  │
│ mode. A farmer-centric digital infrastructure has ensured that benefits reach all farmers across the country without any       │
│ involvement of intermediaries. The Government of India has disbursed over Rs 3.68 lakh crore since inception, with an amount   │
│ of Rs. 1555.52 crore released to beneficiaries in Shahjahanpur District as of now.                                             │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Model: qwen2.5:3b | Tokens: 5308 | Latency: 114329ms | Sources: 18-4-3963, 18-4-1991, 18-3-2388, 18-2-1294, 18-2-1678
(.venv) PS E:\audit2> retrieve query "Malaria eradication"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 7107.94it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: Malaria eradication

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 6145.12it/s]
                                             Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry                  ┃                  Score ┃ Question (excerpt)                   ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-3-3132 │ HEALTH AND FAMILY WELFARE │ -3.6889 (dense: 0.280) │ GOVERNMENT OF INDIA                  │
│     │           │                           │                        │ MINISTRY OF HEALTH AND FAMILYWELFARE │
│     │           │                           │                        │ ...                                  │
├─────┼───────────┼───────────────────────────┼────────────────────────┼──────────────────────────────────────┤
│ 2   │ 18-6-2110 │ LAW AND JUSTICE           │                -4.3776 │ GOVERNMENT OF INDIA                  │
│     │           │                           │                        │ MINISTRY OF LAW AND JUSTICE          │
│     │           │                           │                        │ LEGISLA...                           │
├─────┼───────────┼───────────────────────────┼────────────────────────┼──────────────────────────────────────┤
│ 3   │ 18-2-0711 │ HEALTH AND FAMILY WELFARE │ -6.1383 (dense: 0.275) │ GOVERNMENT OF INDIA                  │
│     │           │                           │                        │ MINISTRY OF HEALTH AND FAMILYWELFARE │
│     │           │                           │                        │ ...                                  │
├─────┼───────────┼───────────────────────────┼────────────────────────┼──────────────────────────────────────┤
│ 4   │ 18-2-0907 │ HEALTH AND FAMILY WELFARE │ -6.8940 (dense: 0.291) │ GOVERNMENT OF INDIA                  │
│     │           │                           │                        │ MINISTRY OF HEALTH AND FAMILYWELFARE │
│     │           │                           │                        │ ...                                  │
├─────┼───────────┼───────────────────────────┼────────────────────────┼──────────────────────────────────────┤
│ 5   │ 18-6-2226 │ AYUSH                     │ -7.4263 (dense: 0.223) │ 1                                    │
│     │           │                           │                        │                                      │
│     │           │                           │                        │ GOVERNMENT OF INDIA                  │
│     │           │                           │                        │ MINISTRY OF AYUSH                    │
│     │           │                           │                        │                                      │
│     │           │                           │                        │ LOK SABHA ...                        │
└─────┴───────────┴───────────────────────────┴────────────────────────┴──────────────────────────────────────┘

Retrieval: 9734ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 22,703 chars | Estimated tokens: 5,675
✓ Prompt size is well within effective context window (5,675 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
╭─────────────────────────────────────────────────────── Generated Answer ───────────────────────────────────────────────────────╮
│ The provided context does not contain sufficient information to answer this question directly regarding malaria eradication.   │
│ The context primarily discusses initiatives related to sickle cell anaemia, pharmacovigilance for Ayush drugs, and             │
│ surveillance platforms in India's health sector. There is no specific mention of malaria eradication efforts or policies.      │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Model: qwen2.5:3b | Tokens: 5758 | Latency: 65359ms | Sources: 18-3-3132, 18-6-2110, 18-2-0711, 18-2-0907, 18-6-2226
(.venv) PS E:\audit2> retreive query "Tuberculosis control programme"
retreive : The term 'retreive' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the
spelling of the name, or if a path was included, verify that the path is correct and try again.
At line:1 char:1

+ retreive query "Tuberculosis control programme"
+ ```
    + CategoryInfo          : ObjectNotFound: (retreive:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException
  ```

(.venv) PS E:\audit2> retrieve query "Tuberculosis control programme"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 6800.79it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: Tuberculosis control programme

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 5819.71it/s]
                                             Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry                  ┃                  Score ┃ Question (excerpt)                   ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-5-2245 │ HEALTH AND FAMILY WELFARE │ -1.4016 (dense: 0.309) │ GOVERNMENT OF INDIA                  │
│     │           │                           │                        │ MINISTRY OF HEALTH AND FAMILYWELFARE │
│     │           │                           │                        │ ...                                  │
├─────┼───────────┼───────────────────────────┼────────────────────────┼──────────────────────────────────────┤
│ 2   │ 18-4-3458 │ HEALTH AND FAMILY WELFARE │ -2.1324 (dense: 0.406) │ GOVERNMENT OF INDIA                  │
│     │           │                           │                        │ MINISTRY OF HEALTH AND FAMILYWELFARE │
│     │           │                           │                        │ ...                                  │
├─────┼───────────┼───────────────────────────┼────────────────────────┼──────────────────────────────────────┤
│ 3   │ 18-3-3132 │ HEALTH AND FAMILY WELFARE │ -2.8960 (dense: 0.358) │ GOVERNMENT OF INDIA                  │
│     │           │                           │                        │ MINISTRY OF HEALTH AND FAMILYWELFARE │
│     │           │                           │                        │ ...                                  │
├─────┼───────────┼───────────────────────────┼────────────────────────┼──────────────────────────────────────┤
│ 4   │ 18-2-3036 │ HEALTH AND FAMILY WELFARE │ -5.9091 (dense: 0.281) │ GOVERNMENT OF INDIA                  │
│     │           │                           │                        │ MINISTRY OF HEALTH AND FAMILYWELFARE │
│     │           │                           │                        │ ...                                  │
├─────┼───────────┼───────────────────────────┼────────────────────────┼──────────────────────────────────────┤
│ 5   │ 18-5-3274 │ HEALTH AND FAMILY WELFARE │ -6.1415 (dense: 0.313) │ GOVERNMENT OF INDIA                  │
│     │           │                           │                        │ MINISTRY OF HEALTH AND FAMILYWELFARE │
│     │           │                           │                        │ ...                                  │
└─────┴───────────┴───────────────────────────┴────────────────────────┴──────────────────────────────────────┘

Retrieval: 9547ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 21,523 chars | Estimated tokens: 5,380
✓ Prompt size is well within effective context window (5,380 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
╭─────────────────────────────────────────────────────── Generated Answer ───────────────────────────────────────────────────────╮
│ The provided context does not contain sufficient information to answer this question directly. The context discusses various   │
│ aspects of healthcare and disease control, including the National Health Mission (NHM), Ayushman Bharat, and Cancer Super      │
│ Speciality Hospital in Bihar, but it does not provide specific details about the Tuberculosis Control Programme.               │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Model: qwen2.5:3b | Tokens: 5462 | Latency: 67422ms | Sources: 18-5-2245, 18-4-3458, 18-3-3132, 18-2-3036, 18-5-3274
(.venv) PS E:\audit2> retrieve query "Ayushman Bharat"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 4748.44it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: Ayushman Bharat

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 4181.47it/s]
                                            Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry                  ┃                 Score ┃ Question (excerpt)                   ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-3-3066 │ HEALTH AND FAMILY WELFARE │ 5.5001 (dense: 0.446) │ GOVERNMENT OF INDIA                  │
│     │           │                           │                       │ MINISTRY OF HEALTH AND FAMILYWELFARE │
│     │           │                           │                       │ ...                                  │
├─────┼───────────┼───────────────────────────┼───────────────────────┼──────────────────────────────────────┤
│ 2   │ 18-2-0173 │ HEALTH AND FAMILY WELFARE │ 5.4760 (dense: 0.423) │ GOVERNMENT OF INDIA                  │
│     │           │                           │                       │ MINISTRY OF HEALTH AND FAMILYWELFARE │
│     │           │                           │                       │ ...                                  │
├─────┼───────────┼───────────────────────────┼───────────────────────┼──────────────────────────────────────┤
│ 3   │ 18-2-1861 │ AYUSH                     │ 5.1504 (dense: 0.580) │ 1                                    │
│     │           │                           │                       │                                      │
│     │           │                           │                       │ GOVERNMENT OF INDIA                  │
│     │           │                           │                       │ MINISTRY OF AYUSH                    │
│     │           │                           │                       │                                      │
│     │           │                           │                       │ LOK SABHA ...                        │
├─────┼───────────┼───────────────────────────┼───────────────────────┼──────────────────────────────────────┤
│ 4   │ 18-5-2262 │ AYUSH                     │ 4.9137 (dense: 0.539) │ 1                                    │
│     │           │                           │                       │                                      │
│     │           │                           │                       │ GOVERNMENT OF INDIA                  │
│     │           │                           │                       │ MINISTRY OF AYUSH                    │
│     │           │                           │                       │ LOK SABHA                            │
│     │           │                           │                       │ U...                                 │
├─────┼───────────┼───────────────────────────┼───────────────────────┼──────────────────────────────────────┤
│ 5   │ 18-5-3392 │ HEALTH AND FAMILY WELFARE │ 4.4360 (dense: 0.428) │ GOVERNMENT OF INDIA                  │
│     │           │                           │                       │ MINISTRY OF HEALTHAND FAMILYWELFARE  │
│     │           │                           │                       │ D...                                 │
└─────┴───────────┴───────────────────────────┴───────────────────────┴──────────────────────────────────────┘

Retrieval: 9282ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 21,998 chars | Estimated tokens: 5,499
✓ Prompt size is well within effective context window (5,499 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
Traceback (most recent call last):
  File "E:\audit2\.venv\Lib\site-packages\httpx\_transports\default.py", line 101, in map_httpcore_exceptions
    yield
  File "E:\audit2\.venv\Lib\site-packages\httpx\_transports\default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_sync\connection_pool.py", line 256, in handle_request
    raise exc from None
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_sync\connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_sync\connection.py", line 103, in handle_request
    return self._connection.handle_request(request)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_sync\http11.py", line 136, in handle_request
    raise exc
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_sync\http11.py", line 106, in handle_request
    ) = self._receive_response_headers(**kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_sync\http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_sync\http11.py", line 217, in _receive_event
    data = self._network_stream.read(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_backends\sync.py", line 126, in read
    with map_exceptions(exc_map):
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.2544.0_x64__qbz5n2kfra8p0\Lib\contextlib.py", line 158, in __exit__
    self.gen.throw(typ, value, traceback)
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_exceptions.py", line 14, in map_exceptions
    raise to_exc(exc) from exc
httpcore.ReadTimeout: timed out

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen runpy></frozen>", line 198, in _run_module_as_main
  File "<frozen runpy></frozen>", line 88, in _run_code
  File "E:\audit2\.venv\Scripts\retrieve.exe\__main__.py", line 7, in <module></module>
  File "E:\audit2\.venv\Lib\site-packages\click\core.py", line 1569, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\click\core.py", line 1490, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\click\core.py", line 1970, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\click\core.py", line 1353, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\click\core.py", line 907, in invoke
    return callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\src\retrieval\cli.py", line 185, in query
    gen_result = generator.generate(question, results)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\src\generation\generator.py", line 279, in generate
    response = self.llm_client.generate(
               ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\src\generation\client.py", line 263, in generate
    return self._generate_ollama(prompt, system, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\src\generation\client.py", line 147, in _generate_ollama
    response = client.post(
               ^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpx\_client.py", line 1144, in post
    return self.request(
           ^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpx\_client.py", line 825, in request
    return self.send(request, auth=auth, follow_redirects=follow_redirects)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpx\_client.py", line 914, in send
    response = self._send_handling_auth(
               ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpx\_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpx\_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpx\_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpx\_transports\default.py", line 249, in handle_request
    with map_httpcore_exceptions():
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.2544.0_x64__qbz5n2kfra8p0\Lib\contextlib.py", line 158, in __exit__
    self.gen.throw(typ, value, traceback)
  File "E:\audit2\.venv\Lib\site-packages\httpx\_transports\default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.ReadTimeout: timed out
(.venv) PS E:\audit2> retrieve query "Women and child nutrition"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 7522.43it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: Women and child nutrition

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 5649.52it/s]
                                               Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry                    ┃                 Score ┃ Question (excerpt)                      ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-3-1993 │ WOMEN AND CHILD DEVELOPMENT │ 5.1991 (dense: 0.510) │ GOVERNMENT OF INDIA                     │
│     │           │                             │                       │ MINISTRY OF WOMEN AND CHILD DEVELOPM... │
├─────┼───────────┼─────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 2   │ 18-2-0874 │ WOMEN AND CHILD DEVELOPMENT │ 3.1932 (dense: 0.390) │ GOVERNMENT OF INDIA                     │
│     │           │                             │                       │ MINISTRY OF WOMEN AND CHILD DEVELOPM... │
├─────┼───────────┼─────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 3   │ 18-3-4175 │ WOMEN AND CHILD DEVELOPMENT │ 3.0537 (dense: 0.508) │ GOVERNMENT OF INDIA                     │
│     │           │                             │                       │ MINISTRY OF WOMEN AND CHILD DEVELOPM... │
├─────┼───────────┼─────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 4   │ 18-6-1080 │ WOMEN AND CHILD DEVELOPMENT │ 2.5822 (dense: 0.469) │ GOVERNMENT OF INDIA                     │
│     │           │                             │                       │ MINISTRY OF WOMEN AND CHILD DEVELOPM... │
├─────┼───────────┼─────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 5   │ 18-2-3193 │ WOMEN AND CHILD DEVELOPMENT │ 2.1129 (dense: 0.398) │ GOVERNMENT OF INDIA                     │
│     │           │                             │                       │ MINISTRY OF WOMEN AND CHILD DEVELOPM... │
└─────┴───────────┴─────────────────────────────┴───────────────────────┴─────────────────────────────────────────┘

Retrieval: 8594ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 28,070 chars | Estimated tokens: 7,017
✓ Prompt size is well within effective context window (7,017 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
╭─────────────────────────────────────────────────────── Generated Answer ───────────────────────────────────────────────────────╮
│ The provided context does not contain sufficient information to answer the specific questions related to women and child       │
│ nutrition. However, it provides details about various schemes and initiatives under the Ministry of Women and Child            │
│ Development aimed at improving nutrition in India, particularly focusing on Anganwadi services, Poshan Abhiyaan, and Mission   │
│ Poshan 2.0. For detailed information on specific aspects such as ICDS status, CMAM implementation, or allocation details for   │
│ child nutrition programs, more specific questions would need to be addressed within the provided context.                      │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Model: qwen2.5:3b | Tokens: 7158 | Latency: 102844ms | Sources: 18-3-1993, 18-2-0874, 18-3-4175, 18-6-1080, 18-2-3193



(.venv) PS E:\audit2> retrieve query "Digital India initiatives"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 6611.67it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: Digital India initiatives

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 6021.52it/s]
                                                    Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry                               ┃                 Score ┃ Question (excerpt)                      ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-5-0653 │ ELECTRONICS AND INFORMATION TECHNOLOGY │ 4.7308 (dense: 0.591) │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF ELECTRONICS AND INFORMAT... │
├─────┼───────────┼────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 2   │ 18-6-2834 │ ELECTRONICS AND INFORMATION TECHNOLOGY │ 3.7618 (dense: 0.492) │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF ELECTRONICS AND INFORMAT... │
├─────┼───────────┼────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 3   │ 18-5-4267 │ PANCHAYATI RAJ                         │ 3.6492 (dense: 0.628) │ 1                                       │
│     │           │                                        │                       │                                         │
│     │           │                                        │                       │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF PANCHAYATI RAJ              │
│     │           │                                        │                       │ LOK...                                  │
├─────┼───────────┼────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 4   │ 18-6-2825 │ COMMUNICATIONS                         │ 3.5612 (dense: 0.569) │ 1                                       │
│     │           │                                        │                       │                                         │
│     │           │                                        │                       │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF COMMUNICATIONS              │
│     │           │                                        │                       │ DE...                                   │
├─────┼───────────┼────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 5   │ 18-6-0139 │ TOURISM                                │ 3.4349 (dense: 0.616) │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF TOURISM                     │
│     │           │                                        │                       │                                         │
│     │           │                                        │                       │    LOK SABHA ...                        │
└─────┴───────────┴────────────────────────────────────────┴───────────────────────┴─────────────────────────────────────────┘

Retrieval: 8344ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 22,479 chars | Estimated tokens: 5,619
✓ Prompt size is well within effective context window (5,619 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
╭─────────────────────────────────────────────────────── Generated Answer ───────────────────────────────────────────────────────╮
│ The provided context does not contain sufficient information to answer this question directly about Digital India initiatives. │
│ The available answers pertain to specific questions related to the Ministry of Electronics and Information Technology,         │
│ Panchayati Raj, Communications, and Tourism. There is no comprehensive overview or detailed list of all Digital India          │
│ initiatives in the given context.                                                                                              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Model: qwen2.5:3b | Tokens: 5711 | Latency: 78594ms | Sources: 18-5-0653, 18-6-2834, 18-5-4267, 18-6-2825, 18-6-0139
(.venv) PS E:\audit2> retrieve query "Ethanol blending programme"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 6864.76it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: Ethanol blending programme

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 6054.65it/s]
                                                      Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry                                 ┃                  Score ┃ Question (excerpt)                       ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-2-2832 │ PETROLEUM AND NATURAL GAS                │  5.8052 (dense: 0.612) │ LOK SABHA                                │
│     │           │                                          │                        │ UNSTARRED QUESTION No. 2832              │
│     │           │                                          │                        │ TO BE ANSWERED ON...                     │
├─────┼───────────┼──────────────────────────────────────────┼────────────────────────┼──────────────────────────────────────────┤
│ 2   │ 18-3-3843 │ CONSUMER AFFAIRS, FOOD AND PUBLIC        │  3.5867 (dense: 0.557) │ ORIGINAL IN HINDI                        │
│     │           │ DISTRIBUTION                             │                        │ GOVERNMENT OF INDIA                      │
│     │           │                                          │                        │ MINISTRY OF CONSUME...                   │
├─────┼───────────┼──────────────────────────────────────────┼────────────────────────┼──────────────────────────────────────────┤
│ 3   │ 18-3-3977 │ PETROLEUM AND NATURAL GAS                │  1.5112 (dense: 0.366) │ LOK SABHA                                │
│     │           │                                          │                        │ UNSTARRED QUESTION No. 3977              │
│     │           │                                          │                        │ TO BE ANSWERED ON...                     │
├─────┼───────────┼──────────────────────────────────────────┼────────────────────────┼──────────────────────────────────────────┤
│ 4   │ 18-4-1610 │ PETROLEUM AND NATURAL GAS                │ -1.5845 (dense: 0.215) │ LOK SABHA                                │
│     │           │                                          │                        │  UNSTARRED QUESTION No. 1610             │
│     │           │                                          │                        │ TO BE ANSWERED ...                       │
├─────┼───────────┼──────────────────────────────────────────┼────────────────────────┼──────────────────────────────────────────┤
│ 5   │ 18-6-0515 │ CONSUMER AFFAIRS, FOOD AND PUBLIC        │ -2.7457 (dense: 0.443) │ GOVERNMENT OF INDIA                      │
│     │           │ DISTRIBUTION                             │                        │ MINISTRY OF CONSUMER AFFAIRS, FOOD & ... │
└─────┴───────────┴──────────────────────────────────────────┴────────────────────────┴──────────────────────────────────────────┘

Retrieval: 9812ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 20,831 chars | Estimated tokens: 5,207
✓ Prompt size is well within effective context window (5,207 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
╭─────────────────────────────────────────────────────── Generated Answer ───────────────────────────────────────────────────────╮
│ The provided context does not contain sufficient information to answer this question directly. The context discusses various   │
│ aspects related to the Ethanol Blended Petrol (EBP) Programme, including adoption rates, number of outlets, foreign exchange   │
│ savings, and other initiatives but does not provide specific details about the Ethanol Blending Programme itself.              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Model: qwen2.5:3b | Tokens: 5290 | Latency: 74875ms | Sources: 18-2-2832, 18-3-3843, 18-3-3977, 18-4-1610, 18-6-0515
(.venv) PS E:\audit2> retrieve query "Renewable energy targets"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 6351.64it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: Renewable energy targets

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 6546.83it/s]
                                              Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry                  ┃                 Score ┃ Question (excerpt)                      ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-6-1667 │ NEW AND RENEWABLE ENERGY  │ 3.0731 (dense: 0.521) │ GOVERNMENT OF INDIA                     │
│     │           │                           │                       │ MINISTRY OF NEW AND RENEWABLE ENERGY... │
├─────┼───────────┼───────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 2   │ 18-5-4660 │ PETROLEUM AND NATURAL GAS │ 2.5780 (dense: 0.431) │ Page 1 of 2                             │
│     │           │                           │                       │                                         │
│     │           │                           │                       │ LOK SABHA                               │
│     │           │                           │                       │ UNSTARRED QUESTION NO. 4660             │
│     │           │                           │                       │ TO...                                   │
├─────┼───────────┼───────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 3   │ 18-3-1418 │ NEW AND RENEWABLE ENERGY  │ 2.2972 (dense: 0.493) │ GOVERNMENT OF INDIA                     │
│     │           │                           │                       │ MINISTRY OF NEW AND RENEWABLE ENERGY... │
├─────┼───────────┼───────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 4   │ 18-4-3022 │ NEW AND RENEWABLE ENERGY  │ 1.7544 (dense: 0.514) │ GOVERNMENT OF INDIA                     │
│     │           │                           │                       │ MINISTRY OF NEW AND RENEWABLE ENERGY... │
├─────┼───────────┼───────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 5   │ 18-4-2128 │ NEW AND RENEWABLE ENERGY  │ 1.4670 (dense: 0.508) │ GOVERNMENT OF INDIA                     │
│     │           │                           │                       │ MINISTRY OF NEW AND RENEWABLE ENERGY... │
└─────┴───────────┴───────────────────────────┴───────────────────────┴─────────────────────────────────────────┘

Retrieval: 9812ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 25,306 chars | Estimated tokens: 6,326
✓ Prompt size is well within effective context window (6,326 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
╭─────────────────────────────────────────────────────── Generated Answer ───────────────────────────────────────────────────────╮
│ The Production Linked Incentive (PLI) Scheme for High Efficiency Solar PV Modules aims to promote manufacturing of high        │
│ efficiency solar PV modules in India and thus reduce import dependency in the area of renewable energy. Under PLI Scheme for   │
│ High Efficiency Solar PV Modules, Letters of Award have been issued for setting up of 48,337 MW of fully/partially integrated  │
│ solar PV module manufacturing units. As on date, the solar PV module manufacturing capacity in the country as per Approved     │
│ List of Models & Manufacturers (ALMM) list for solar PV modules is around 63 GW.                                               │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Model: qwen2.5:3b | Tokens: 6462 | Latency: 79360ms | Sources: 18-6-1667, 18-5-4660, 18-3-1418, 18-4-3022, 18-4-2128
(.venv) PS E:\audit2> retrieve query "Fisheries development"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 4960.48it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: Fisheries development

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 4329.13it/s]
                                                     Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry                                 ┃                 Score ┃ Question (excerpt)                      ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-4-1255 │ FISHERIES, ANIMAL HUSBANDRY AND DAIRYING │ 3.6616 (dense: 0.418) │ GOVERNMENT OF INDIA                     │
│     │           │                                          │                       │ MINISTRY OF FISHERIES, ANIMAL HUSBAN... │
├─────┼───────────┼──────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 2   │ 18-3-2343 │ FISHERIES, ANIMAL HUSBANDRY AND DAIRYING │ 3.5546 (dense: 0.511) │ GOVERNMENT OF INDIA                     │
│     │           │                                          │                       │ MINISTRY OF FISHERIES, ANIMAL HUSBAN... │
├─────┼───────────┼──────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 3   │ 18-4-0411 │ FISHERIES, ANIMAL HUSBANDRY AND DAIRYING │ 3.3557 (dense: 0.517) │ GOVERNMENT OF INDIA                     │
│     │           │                                          │                       │ MINISTRY OF FISHERIES, ANIMAL HUSBAN... │
├─────┼───────────┼──────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 4   │ 18-5-0291 │ FISHERIES, ANIMAL HUSBANDRY AND DAIRYING │ 3.3222 (dense: 0.528) │ GOVERNMENT OF INDIA                     │
│     │           │                                          │                       │ MINISTRY OF FISHERIES, ANIMAL HUSBAN... │
├─────┼───────────┼──────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 5   │ 18-5-2570 │ FISHERIES, ANIMAL HUSBANDRY AND DAIRYING │ 2.4414 (dense: 0.542) │ GOVERNMENT OF INDIA                     │
│     │           │                                          │                       │ MINISTRY OF FISHERIES, ANIMAL HUSBAN... │
└─────┴───────────┴──────────────────────────────────────────┴───────────────────────┴─────────────────────────────────────────┘

Retrieval: 9734ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 24,328 chars | Estimated tokens: 6,082
✓ Prompt size is well within effective context window (6,082 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
╭─────────────────────────────────────────────────────── Generated Answer ───────────────────────────────────────────────────────╮
│ The provided context does not contain sufficient information to answer this question directly. The context primarily discusses │
│ the Pradhan Mantri Matsya Sampada Yojana (PMMSY) and related projects in various states, as well as details about the          │
│ Rashtriya Gokul Mission and Kamdhenu. There is no specific focus on fisheries development that would allow for a detailed      │
│ answer regarding fish production, infrastructure development, or other aspects of fisheries management.                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Model: qwen2.5:3b | Tokens: 6192 | Latency: 120281ms | Sources: 18-4-1255, 18-3-2343, 18-4-0411, 18-5-0291, 18-5-2570
(.venv) PS E:\audit2> retrieve query "Visa on arrival"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 3620.93it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: Visa on arrival

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 3379.02it/s]
                                    Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry         ┃                  Score ┃ Question (excerpt)           ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-5-0219 │ TOURISM          │ -2.2999 (dense: 0.249) │ GOVERNMENT OF INDIA          │
│     │           │                  │                        │ MINISTRY OF TOURISM          │
│     │           │                  │                        │                              │
│     │           │                  │                        │ LOK SABHA                    │
│     │           │                  │                        │ UN...                        │
├─────┼───────────┼──────────────────┼────────────────────────┼──────────────────────────────┤
│ 2   │ 18-3-0143 │ TOURISM          │ -2.7521 (dense: 0.260) │ GOVERNMENT OF INDIA          │
│     │           │                  │                        │ MINISTRY OF TOURISM          │
│     │           │                  │                        │                              │
│     │           │                  │                        │ LOK SABHA                    │
│     │           │                  │                        │ U...                         │
├─────┼───────────┼──────────────────┼────────────────────────┼──────────────────────────────┤
│ 3   │ 18-5-2106 │ EXTERNAL AFFAIRS │ -3.0880 (dense: 0.402) │ 1                            │
│     │           │                  │                        │ GOVERNMENT OF INDIA          │
│     │           │                  │                        │ MINISTRY OF EXTERNAL AFFAIRS │
│     │           │                  │                        │ LOK SA...                    │
├─────┼───────────┼──────────────────┼────────────────────────┼──────────────────────────────┤
│ 4   │ 18-4-3516 │ EXTERNAL AFFAIRS │ -3.2105 (dense: 0.179) │ 1                            │
│     │           │                  │                        │ GOVERNMENT OF INDIA          │
│     │           │                  │                        │ MINISTRY OF EXTERNAL AFFAIRS │
│     │           │                  │                        │ LOK SA...                    │
├─────┼───────────┼──────────────────┼────────────────────────┼──────────────────────────────┤
│ 5   │ 18-3-3115 │ EXTERNAL AFFAIRS │ -3.2883 (dense: 0.275) │ GOVERNMENT OF INDIA          │
│     │           │                  │                        │ MINISTRY OF EXTERNAL AFFAIRS │
│     │           │                  │                        │ LOK SABH...                  │
└─────┴───────────┴──────────────────┴────────────────────────┴──────────────────────────────┘

Retrieval: 12515ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 27,729 chars | Estimated tokens: 6,932
✓ Prompt size is well within effective context window (6,932 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
╭─────────────────────────────────────────────────────── Generated Answer ───────────────────────────────────────────────────────╮
│ The provided context does not contain sufficient information to answer this question specifically about the Visa on Arrival    │
│ facility. The context discusses visa-related matters in general, but does not provide details about the Visa on Arrival        │
│ facility for tourists.                                                                                                         │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Model: qwen2.5:3b | Tokens: 6991 | Latency: 87515ms | Sources: 18-5-0219, 18-3-0143, 18-5-2106, 18-4-3516, 18-3-3115
(.venv) PS E:\audit2> retrieve query "Skill India programme"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 3623.42it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: Skill India programme

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 3971.82it/s]
                                                    Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry                               ┃                 Score ┃ Question (excerpt)                      ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-2-0117 │ SKILL DEVELOPMENT AND ENTREPRENEURSHIP │ 7.0434 (dense: 0.652) │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF SKILL DEVELOPMENT AND EN... │
├─────┼───────────┼────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 2   │ 18-4-0081 │ SKILL DEVELOPMENT AND ENTREPRENEURSHIP │ 6.0251 (dense: 0.595) │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF SKILL DEVELOPMENT AND EN... │
├─────┼───────────┼────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 3   │ 18-4-0060 │ SKILL DEVELOPMENT AND ENTREPRENEURSHIP │ 5.9089 (dense: 0.618) │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF SKILL DEVELOPMENT AND EN... │
├─────┼───────────┼────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 4   │ 18-6-1245 │ SKILL DEVELOPMENT AND ENTREPRENEURSHIP │ 5.4451 (dense: 0.529) │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF SKILL DEVELOPMENT AND EN... │
├─────┼───────────┼────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 5   │ 18-3-0019 │ SKILL DEVELOPMENT AND ENTREPRENEURSHIP │ 5.2018 (dense: 0.616) │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF SKILL DEVELOPMENT AND EN... │
└─────┴───────────┴────────────────────────────────────────┴───────────────────────┴─────────────────────────────────────────┘

Retrieval: 9094ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 26,503 chars | Estimated tokens: 6,625
✓ Prompt size is well within effective context window (6,625 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
Traceback (most recent call last):
  File "E:\audit2\.venv\Lib\site-packages\httpx\_transports\default.py", line 101, in map_httpcore_exceptions
    yield
  File "E:\audit2\.venv\Lib\site-packages\httpx\_transports\default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_sync\connection_pool.py", line 256, in handle_request
    raise exc from None
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_sync\connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_sync\connection.py", line 103, in handle_request
    return self._connection.handle_request(request)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_sync\http11.py", line 136, in handle_request
    raise exc
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_sync\http11.py", line 106, in handle_request
    ) = self._receive_response_headers(**kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_sync\http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_sync\http11.py", line 217, in _receive_event
    data = self._network_stream.read(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_backends\sync.py", line 126, in read
    with map_exceptions(exc_map):
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.2544.0_x64__qbz5n2kfra8p0\Lib\contextlib.py", line158, in __exit__
    self.gen.throw(typ, value, traceback)
  File "E:\audit2\.venv\Lib\site-packages\httpcore\_exceptions.py", line 14, in map_exceptions
    raise to_exc(exc) from exc
httpcore.ReadTimeout: timed out

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen runpy></frozen>", line 198, in _run_module_as_main
  File "<frozen runpy></frozen>", line 88, in _run_code
  File "E:\audit2\.venv\Scripts\retrieve.exe\__main__.py", line 7, in <module></module>
  File "E:\audit2\.venv\Lib\site-packages\click\core.py", line 1569, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\click\core.py", line 1490, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\click\core.py", line 1970, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\click\core.py", line 1353, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\click\core.py", line 907, in invoke
    return callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\src\retrieval\cli.py", line 185, in query
    gen_result = generator.generate(question, results)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\src\generation\generator.py", line 279, in generate
    response = self.llm_client.generate(
               ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\src\generation\client.py", line 263, in generate
    return self._generate_ollama(prompt, system, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\src\generation\client.py", line 147, in _generate_ollama
    response = client.post(
               ^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpx\_client.py", line 1144, in post
    return self.request(
           ^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpx\_client.py", line 825, in request
    return self.send(request, auth=auth, follow_redirects=follow_redirects)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpx\_client.py", line 914, in send
    response = self._send_handling_auth(
               ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpx\_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpx\_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpx\_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\audit2\.venv\Lib\site-packages\httpx\_transports\default.py", line 249, in handle_request
    with map_httpcore_exceptions():
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.2544.0_x64__qbz5n2kfra8p0\Lib\contextlib.py", line158, in __exit__
    self.gen.throw(typ, value, traceback)
  File "E:\audit2\.venv\Lib\site-packages\httpx\_transports\default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.ReadTimeout: timed out
(.venv) PS E:\audit2> retrieve query "MSME credit support"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 3962.99it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: MSME credit support

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 3344.58it/s]
                                                  Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry                            ┃                 Score ┃ Question (excerpt)                     ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-4-5298 │ MICRO, SMALL AND MEDIUM ENTERPRISES │ 5.1135 (dense: 0.513) │ GOVERNMENT OF INDIA                    │
│     │           │                                     │                       │                                        │
│     │           │                                     │                       │ MINISTRY OF MICRO SMALL AND MEDIUM ... │
├─────┼───────────┼─────────────────────────────────────┼───────────────────────┼────────────────────────────────────────┤
│ 2   │ 18-6-1909 │ MICRO, SMALL AND MEDIUM ENTERPRISES │ 3.9986 (dense: 0.519) │ GOVERNMENT                             │
│     │           │                                     │                       │                                        │
│     │           │                                     │                       │ OF                                     │
│     │           │                                     │                       │                                        │
│     │           │                                     │                       │ INDIA                                  │
│     │           │                                     │                       │                                        │
│     │           │                                     │                       │ MINISTRY                               │
│     │           │                                     │                       │                                        │
│     │           │                                     │                       │ OF                                     │
│     │           │                                     │                       │                                        │
│     │           │                                     │                       │ MICRO                                  │
│     │           │                                     │                       │ , SMALL                                │
│     │           │                                     │                       │  ...                                   │
├─────┼───────────┼─────────────────────────────────────┼───────────────────────┼────────────────────────────────────────┤
│ 3   │ 18-5-0853 │ MICRO, SMALL AND MEDIUM ENTERPRISES │ 3.6675 (dense: 0.450) │ GOVERNMENT OF INDIA                    │
│     │           │                                     │                       │                                        │
│     │           │                                     │                       │ MINISTRY OF MICRO, SMALL AND MEDIUM... │
├─────┼───────────┼─────────────────────────────────────┼───────────────────────┼────────────────────────────────────────┤
│ 4   │ 18-4-3235 │ MICRO, SMALL AND MEDIUM ENTERPRISES │ 2.7814 (dense: 0.413) │ GOVERNMENT OF INDIA                    │
│     │           │                                     │                       │                                        │
│     │           │                                     │                       │ MINISTRY OF MICRO, SMALL AND MEDIUM... │
├─────┼───────────┼─────────────────────────────────────┼───────────────────────┼────────────────────────────────────────┤
│ 5   │ 18-4-1612 │ FINANCE                             │ 1.9928 (dense: 0.548) │ GOVERNMENT OF INDIA                    │
│     │           │                                     │                       │ MINISTRY OF FINANCE                    │
│     │           │                                     │                       │ DEPARTMENT OF F...                     │
└─────┴───────────┴─────────────────────────────────────┴───────────────────────┴────────────────────────────────────────┘

Retrieval: 8437ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 21,881 chars | Estimated tokens: 5,470
✓ Prompt size is well within effective context window (5,470 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
╭─────────────────────────────────────────────────────── Generated Answer ───────────────────────────────────────────────────────╮
│ The Ministry of Micro, Small and Medium Enterprises (MSME) has implemented several measures to strengthen access to credit and │
│ equity for MSMEs, including the following:                                                                                     │
│                                                                                                                                │
│ - The Emergency Credit Line Guarantee Scheme (ECLGS), launched in May 2020 as part of Aatmanirbhar Bharat Package, provided    │
│ guarantees covering a value of Rs. 2.42 lakh crore to MSMEs by March 31, 2023.                                                 │
│                                                                                                                                │
│ - The Credit Guarantee Scheme (CGS) for MSEs through the Credit Guarantee Fund Trust for Micro and Small Enterprises (CGTMSE)  │
│ provides credit guarantee for loans extended by Member Lending Institutions to new as well as existing MSEs without collateral │
│ security or third-party guarantees. The scope of the Scheme was expanded in April 2023 with a corpus infusion of Rs. 9,000     │
│ crore into CGTMSE and Standard Rate of Annual Guarantee Fee (AGF) reduced by 50%, to as low as 0.37% per annum.                │
│                                                                                                                                │
│ - Self-Reliant India (SRI) Fund has been set up to infuse Rs. 50,000 crore as equity funding in MSMEs with a provision of Rs.  │
│ 10,000 crore from the Government of India and Rs. 40,000 crore through Private Equity/Venture Capital Funds.                   │
│                                                                                                                                │
│ - The Prime Minister's Employment Generation Programme provides Margin Money subsidy up to 35% for setting up new micro        │
│ enterprises in the non-farm sector with a project cost of Rs. 50 lakh for manufacturing and Rs. 20 lakh for service            │
│ enterprises.                                                                                                                   │
│                                                                                                                                │
│ - PM Vishwakarma Scheme was launched on September 17, 2023, providing end-to-end holistic support to artisans and craftspeople │
│ from 18 traditional trades who work with their hands and tools, including provision of loans up to Rs. 3 lakh with interest    │
│ subvention of max upto 8%.                                                                                                     │
│                                                                                                                                │
│ - The Government conducts regular review meetings to track the implementation and assess the performance of its schemes, and   │
│ holds stakeholder consultations on a regular basis for feedback and necessary policy interventions as required.                │
│                                                                                                                                │
│ Additionally, other initiatives across the country include:                                                                    │
│                                                                                                                                │
│ - The International Cooperation Scheme of MSME provides financial assistance to MSMEs for participation in international trade │
│ fairs and exhibitions.                                                                                                         │
│ - The Directorate General of Foreign Trade (DGFT) is piloting E-commerce Export Hubs (ECEHs) to assist MSMEs and               │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Model: qwen2.5:3b | Tokens: 6011 | Latency: 117000ms | Sources: 18-4-5298, 18-6-1909, 18-5-0853, 18-4-3235, 18-4-1612
(.venv) PS E:\audit2> retrieve query "Startup India"
Loading pipeline from storage\hybrid_rag...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 3388.42it/s]
✓ Loaded Hybrid RAG pipeline from storage\hybrid_rag

Question: Startup India

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 3938.52it/s]
                                                    Retrieved Q&A Records
┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Doc ID    ┃ Ministry                               ┃                 Score ┃ Question (excerpt)                      ┃
┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ 18-5-4302 │ COMMERCE AND INDUSTRY                  │ 6.9968 (dense: 0.631) │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF COMMERCE & INDUSTRY         │
│     │           │                                        │                       │ DEP...                                  │
├─────┼───────────┼────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 2   │ 18-3-3626 │ COMMERCE AND INDUSTRY                  │ 5.9469 (dense: 0.494) │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF COMMERCE & INDUSTRY         │
│     │           │                                        │                       │ DEP...                                  │
├─────┼───────────┼────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 3   │ 18-3-2321 │ COMMERCE AND INDUSTRY                  │ 5.4086 (dense: 0.607) │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF COMMERCE & INDUSTRY         │
│     │           │                                        │                       │ DEP...                                  │
├─────┼───────────┼────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 4   │ 18-4-0929 │ SKILL DEVELOPMENT AND ENTREPRENEURSHIP │ 5.4015 (dense: 0.432) │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF SKILL DEVELOPMENT AND EN... │
├─────┼───────────┼────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────┤
│ 5   │ 18-2-0214 │ AGRICULTURE AND FARMERS WELFARE        │ 2.3963 (dense: 0.468) │ GOVERNMENT OF INDIA                     │
│     │           │                                        │                       │ MINISTRY OF AGRICULTURE AND FARMERS ... │
└─────┴───────────┴────────────────────────────────────────┴───────────────────────┴─────────────────────────────────────────┘

Retrieval: 8937ms total

Generating answer with qwen2.5:3b...
[Generation Audit] Prompt size: 24,780 chars | Estimated tokens: 6,195
✓ Prompt size is well within effective context window (6,195 / 8,192 tokens).
✓ Saved exact prompt to 'generation_prompt_debug.txt' for audit.
╭─────────────────────────────────────────────────────── Generated Answer ───────────────────────────────────────────────────────╮
│ The provided context does not contain sufficient information to answer this question directly about Startup India initiatives. │
│ The context discusses various aspects related to startups but does not provide specific details on the number of startups      │
│ established or registered, their status (active/inactive/closed), reasons for closure, steps taken by the government to ensure │
│ sustainability and long-term success of startups, nor detailed information about benefits or measures towards startups in tier │
│ II and III cities.                                                                                                             │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Model: qwen2.5:3b | Tokens: 6318 | Latency: 78109ms | Sources: 18-5-4302, 18-3-3626, 18-3-2321, 18-4-0929, 18-2-0214
(.venv) PS E:\audit2>
