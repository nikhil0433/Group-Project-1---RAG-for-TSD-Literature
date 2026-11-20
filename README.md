# TSD Literature Research Assistant

**Course:** Project 1 — Language Models and Agentic Workflows  
**Notebook:** `TSD_Project1.ipynb`  
**Recommended environment:** Google Colab

## About the project

I built this project to answer questions from Traffic Speed Deflectometer (TSD) literature while keeping the supporting evidence easy to check. Instead of generating an answer from the language model alone, the system searches a collection of TSD publications and connects its claims to passages from the source PDFs.

The main experiment compares two workflows using the same documents, retrieval index, and language model:

- **Single-pass RAG:** Searches once, selects relevant passages, and generates a response. It does not run an internal semantic verification or revision loop.
- **Adaptive agent:** Plans what needs to be answered, searches the literature, and uses the retrieved evidence to decide what to do next. It can search again, check a draft against its evidence, revise the draft, ask for clarification, or abstain when evidence is insufficient.

For retrieval, I combined BM25 with `all-MiniLM-L6-v2` embeddings using reciprocal rank fusion (RRF). The model selects evidence IDs for its claims, but Python checks the references and extracts the quotations directly from the original passages. This keeps citation construction separate from the model's judgment about what a source means.

## Files and inputs

To run the notebook, you need:

1. `TSD_Project1.ipynb` — the complete implementation and evaluation notebook.
2. A ZIP file containing the **34 TSD-related PDF documents** used to build the literature collection. The notebook checks that the archive contains 34 PDFs.
3. Access to the configured course model endpoint and its API key.

The **30 evaluation questions and reference-answer components are already included in the notebook**. You do not need to upload the ground-truth Excel workbook to run this version.

The recorded experiment used 34 PDFs containing 3,405 physical pages. After text extraction and chunking, the searchable index contained 4,881 passages.

## Dependencies

The notebook installs the required packages in its first code cell. The main dependencies are:

```text
PyMuPDF==1.26.7
pydantic>=2.8,<3
numpy>=1.26,<3
pandas>=2.0,<3
scikit-learn>=1.4,<2
sentence-transformers>=3.0,<6
openai>=1.60,<3
ipython>=8
```

Google Colab is the easiest way to run the project. A GPU is optional, although it can help when MiniLM generates embeddings for the first time. Internet access is needed to install packages, download the embedding model if it is not cached, and make the course API calls.

## How to run the notebook

1. Open [Google Colab](https://colab.research.google.com/) and upload `TSD_Project1.ipynb` using **File → Upload notebook**.
2. Optionally select a GPU under **Runtime → Change runtime type**. The first index build may be slower on CPU.
3. Add the course API key to **Colab Secrets** under the name `OPENAI_API_KEY`, and enable notebook access. The code can also read it from an environment variable or prompt for it securely.
4. Run the notebook from the top. During setup, upload the ZIP containing the 34 PDFs when prompted.
5. The notebook extracts the PDF text, creates the passages and source IDs, and builds the BM25 and MiniLM retrieval indexes.
6. Run the workflow and evaluation cells in order. They define the tools, the two RAG workflows, the reference questions, and the evaluation procedure.
7. Run the batch evaluation to produce the saved results and traces. Then run the semantic regrading cell to score the already saved responses. **The regrading step does not rerun the 60 question-answering workflows.**

The main code sections are labeled Cells 3, 5, 7, 9, 11, 12, 13, and 14 in the notebook. Keep their original execution order, since later sections use objects created earlier.



### Using a PDF ZIP that is already in Colab

If the input archive is already available, you can set its path before the configuration cell:

```python
import os
os.environ["TSD_ZIP_PATH"] = "/content/TSD_Literature.zip"
```

Replace the example filename with the actual location of your ZIP.

## Retrieval and agent settings

| Item | Configuration |
|---|---|
| Document processing | Page-based chunks of up to 1,800 characters with 250-character overlap |
| Sparse retrieval | BM25 (`k1=1.5`, `b=0.75`) |
| Dense retrieval | `all-MiniLM-L6-v2` (384-dimensional embeddings) |
| Rank fusion | RRF, constant 60 |
| Evidence supplied to model | Up to 8 passages, within 14,000 serialized bytes |
| Read-only tools | `search_documents` and `lookup_evidence` |
| Maximum searches per adaptive run | 2 |
| Maximum revisions after the first draft | 2 |
| Agent iterations | 10 |
| Workflow model calls | 14 |
| Dispatcher tool calls | 70 |
| Time limit | 600 seconds |
| Token budget | 400,000 |
| Estimated cost limit | $1.00 per question |

Tool requests go through a dispatcher that validates their arguments and checks the permitted actions and budgets. The retrieved PDF text is used as evidence, not as instructions that can change the tool permissions.

## Evaluation

I used **30 questions**, with five questions in each category: factual retrieval, comparison, multi-source synthesis, conflicting evidence, ambiguous requests, and unanswerable requests. Each question was run once using both workflows, giving **60 recorded runs**.

The evaluation looks at more than whether the answer sounds reasonable:

- **Behavior correctness:** Did the workflow choose an acceptable action—answer, clarify, or abstain?
- **Semantic quality:** How well did the returned response cover the expected answer components? Each component received `0`, `0.5`, or `1` from the LLM-based semantic judge.
- **Task-level semantic score:** The semantic score is kept only when the behavior is acceptable. An inappropriate action or a run with no final response receives zero. This score is averaged across all 30 questions.
- **Citation integrity:** Do the source IDs and extracted quotations match the indexed passages?
- **Evidence support and efficiency:** Are claims supported by their citations, and how many calls, tokens, seconds, and estimated API dollars did the workflow use?

The original post-run evaluation and the later semantic regrading are separate steps. The semantic regrading uses the saved responses and reference components. It does not generate new answers.

### Results from my completed experiment

| Measure | Single-pass RAG | Adaptive agent |
|---|---:|---:|
| Final responses | 30/30 | 25/30 |
| Behavior correctness | 83.3% (25/30) | 73.3% (22/30) |
| Semantic quality, returned responses only | 47.9% | 52.8% |
| Task-level semantic score, all questions | 44.9% | 40.6% |
| Citation integrity | 97/98 (99.0%) | 72/72 (100.0%) |
| Average execution time | 13.0 s | 37.9 s |
| Average estimated cost/question | $0.00118 | $0.00310 |

The adaptive agent scored slightly higher on semantic quality **among the responses it returned**, but it did not return a final response for five questions. Once missing responses and inappropriate actions were included, the simpler RAG workflow had the higher task-level score. The agent was better at asking for clarification on ambiguous questions; the baseline was better at abstaining on unanswerable ones.

These numbers come from the saved experiment. A new run can produce different outputs because the language model is not fully deterministic. The costs are estimates based on token usage, not verified billing charges. The reported averages exclude the additional cost of the later semantic regrading.

## Results files and execution traces

The notebook saves its output under `tsd_project/` in the working directory (usually `/content/` in Colab). The main exports are:

```text
tsd_project/
├── corpus_manifest.json
├── pages.json
├── chunks.json
├── live_evaluation_v5_strict_semantic_checks/
│   ├── evaluation_results.csv
│   ├── evaluation_summary_by_variant.csv
│   ├── evaluation_summary_by_slice.csv
│   ├── runs/
│   └── representative_traces/
├── TSD_Project1_Results_v5_StrictSemanticChecks.zip
├── semantic_regrade_v1/
│   ├── semantic_summary_by_variant.csv
│   ├── semantic_summary_by_slice.csv
│   └── graded_runs/
└── TSD_Project1_Semantic_Regrade_v1.zip
```

The results ZIP contains the workflow executions, including final responses and traces. The semantic-regrading ZIP contains the component-level grades and task-level scores.

Some useful trace examples are **Q01** (revision budget exhausted), **Q02** (adaptive verification completed), **Q07** (more searching and revision did not improve the scored response), **Q21** (clarification of an ambiguous question), and **Q26** (baseline abstention compared with adaptive clarification).

**Save both ZIP files after the run.** Files stored only in a temporary Colab runtime may be lost when the session disconnects. The original evaluation can resume only if its saved records are still available or have been restored.

## Common issues

**The notebook rejects my ZIP.** Check that it contains exactly 34 PDF files, not an extra folder of unrelated PDFs or a different document count.

**The retrieval setup is slow.** The first run may download MiniLM and create embeddings for all passages. This is usually the slowest setup step, especially on CPU.

**The API call fails.** Confirm that `OPENAI_API_KEY` is accessible and that your course account supports the specified model and Portkey endpoint.

**The adaptive agent returns no final answer.** This can happen if it exhausts its revision budget or hits another stopping condition. It is recorded as a failed completion, not as a correct evidence-based abstention.

**Semantic regrading cannot find the results archive.** Run the evaluation/export cell first, or restore the original results ZIP and update the `SOURCE_ZIP` path in the semantic-regrading cell.

## Reproducibility and submission notes

The code, question set, reference components, and evaluation procedure are in the notebook. The input research PDFs are supplied separately. The saved results and trace archives document what actually happened in the completed run, including unsuccessful agent executions.

For submission, include the notebook, this README, the evaluation results, and selected traces. Remove API keys and any private data from code, notebook outputs, and exported logs. If you share the original research PDFs, check that redistribution is permitted.

The evaluation uses one trial per question and an LLM-based judge rather than independent human scoring of every response, so the numerical results should be interpreted with those limits in mind.

## AI-use disclosure

ChatGPT was used to assist with code editing and debugging during project development.
