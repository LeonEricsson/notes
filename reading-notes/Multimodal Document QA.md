

#### M3DocRAG
is a framework that accelerates at multi-modal multi-document VQA, a RAG framework if you will. M3DocRag is a Visual RAG implementation that uses **ColiPali** to generate multi-modal document embeddings via a late interaction mechanism for each page in the document corpus. Page retrieval is performed using a MaxSim score, in practice this is implemented through a FAISS vector database. The top-k pages are retrieved based on similarity with text query. The top-k retrieved page images are fed into **Qwen2-VL**.

![[Screenshot 2025-11-24 at 18.36.28.png|650]]

The team finds that approximate indexing methods provide a favorable speed accuracy trade-off, with query latency decreasing from 21s to <2s when changin from the naive index with exact search (FlatIP) to inverted file (IVFFlat). By default they use FlatIP+IVFFlat.  

![[Screenshot 2025-11-24 at 18.53.03.png]]
#### M3DocVQA
Focuses on implementing challenging VQA spanning multiple documents, phrased as open-domain questions. This is specifically targeted at multi-modal retrieval across many documents, imitating real-world scenarios where users often seek answers that span across multiple documents and modalities. The dataset contains of 2,441 multi-hop questions spanning 3,368 PDF documents totalling 41k pages. There are barely any evaluated models for the reported results in the paper, but it "beats text RAG" in at least one implementation.

![[Screenshot 2025-11-24 at 18.48.01.png|600]]

**wrt Vägval:** M3DocVQA is not usable since the questions span multiple documents, beyond the available context length of a model. However this benchmark is very interesting for FOI-RAG.

#### MMLongBench
Is a collection of datasets bundled together into a benchmark. The task categories covered by this benchmark are Visual RAG, NIAH, ICL, Summarization and Long-document VQA. This benchmark provides a comprehensive evaluation of visual document understanding across long contexts. For our use case, it is not that interesting, perhaps the Visual RAG components are interesting.

#### MMLongBench-Doc
Is a multi-modal VQA benchmark that focuses on document understanding inside single, i.e closed-domain, documents, long-documents to be exact. The average length of documents in the dataset is 47.5 pages or 21,214 text tokens. The answers to questions rely on evidence from difference sources: text, image, chart, table. 33% of the questions are cross-page questions, 22% of the questions are unanswerable, used to detect hallucinations. **All documents are supplied in PDF-format!** Sampels are clearly marked with their evidence source (TXT, LAY, CHA, TAB, IMG), question type (single-page, cross-page, unanswerable). At this time (late 2024), 12 out of 14 VLMs (except GPT 4o/4V) present inferior performance to their LLM counterparts which rely on lossy OCR through Tesseract. These are the first results I have seen comparing raw VLM to OCR+LLM, however at the time the paper was published VLMs were still very young, since then we've seen the release of very strong VLM models. Their updated leaderboard has some newer VLM models, which beat the OCR+LLM baselines from the paper, but thise baselines are also outdated, so we don't get an updated VLM vs OCR+LLM comparison.

![[Screenshot 2025-11-24 at 19.31.13.png|900]]

#### LongDocURL
Very similar to MMLongBench-Doc. The benchmark categorizes its tasks into Understanding, Locating, and Reasoning, with questions spanning multi-page or single-page. The average page length of PDFs in LongDocURL is significantly longer than in MMLongBench-Doc, 85.6 pages on average, totalling 43622 tokens. This benchmark also clearly labels evidence type (pure-text, layout, table, figure). From what I can gather, this benchmark also supplies source PDFs. Similarly to MMLongBench, the paper reports results comparing OCR+LLM and pure VLMs. Surprisingly, on LongDocURL, OCR+LLM significantly trails VLM performance, directly contradicting the results in MMLongBench despite being published roughly the same time and using roughly the same models. 