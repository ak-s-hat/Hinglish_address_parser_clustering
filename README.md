## Indian Address Parsing – End-to-End Workflow

This project builds a custom NER model for Indian address parsing, runs inference to structure raw addresses into fields, and performs clustering for canonicalization and deduplication.
The goal is to segment raw unstructured addresses into structured fields and cluster customer IDs belonging to the same physical household—even with typos, different formats, and multilingual noise.

### What’s inside
- Training‑data generator: `src/bio_tagger.py`
- Model training and evaluation: `src/muril-train (1).ipynb`
- Inference, standardization, FAISS indexing, and clustering: `src/pipeline.ipynb`
- Multi‑model JSON parsing (optional/LLM): `src/multi_model_free_jsonator.py`

---

### Workflow overview 
1) LLM JSON bootstrapping (from raw address strings)
   - Use `src/multi_model_free_jsonator.py` to convert raw addresses into a structured CSV (fields: `house_number`, `road_details`, `pincode`, `complete_address`, ...).
2) Convert to BIO tags (structured CSV → NER dataset)
   - Use `src/bio_tagger.py` on the LLM‑structured CSV to create `sentence` + `tags` BIO data aligned to `complete_address`.
3) Train the custom NER model
   - In `src/muril-train (1).ipynb`, fine‑tune MURIL on the BIO dataset and export the model/tokenizer.
4) Parse + cluster in the pipeline
   - In `src/pipeline.ipynb`, standardize addresses, run the trained NER, post‑process fields, build FAISS indexes (cosine with `IndexFlatIP`), and cluster (threshold grouping, hierarchical; optional token clustering).

---

### 1) Generate structured data from raw addresses (LLM JSON)
Code: `src/multi_model_free_jsonator.py`

Purpose: Convert raw address strings into a consistent structured CSV with the schema fields (`house_number`, `plot_number`, `floor`, `road_details`, `khasra_number`, `block`, `apartment`, `landmark`, `locality`, `area`, `locality2`, `village`, `pincode`, `complete_address`).

Notes:
- Uses Groq LLMs with auto‑failover and token/call throttling.
- Robust JSON extraction (list or block fallback) and column alignment before export.

Output: a structured CSV ready for BIO conversion in step 2.

---

### 2) Convert structured CSV → BIO‑tagged NER dataset
Code: `src/bio_tagger.py`

Purpose: Convert structured rows (from the LLM JSON step or any pre‑labeled source) into tokenized `sentence` plus BIO `tags` for token‑classification training.

How it works (high‑level):
- Cleans and tokenizes `complete_address`, with special handling to keep 6‑digit PIN codes intact.
- Locates spans for each field across tokens; assigns B‑/I‑ tags; non‑entity tokens get `O`.
- Outputs a CSV with two columns: `sentence` and `tags`.

Run it:
1) Provide the LLM‑structured CSV (from step 1) as `input_file`.
2) In `src/bio_tagger.py`, set `input_file` and `output_file` in `__main__` and run the script.
3) Inspect the printed tag stats and samples.

BIO output schema:
- `sentence`: tokenized address string
- `tags`: space‑separated BIO tags aligned with the tokens in `sentence`

Why a custom NER (vs. continuing with LLM JSON in production):
- **Scalability**: Batch NER inference is fast and local; LLM APIs (especially free tiers) are slow and rate‑limited.
- **Cost**: Once trained, inference is inexpensive compared to per‑call LLM usage at scale.
- **Reliability**: LLMs can hallucinate or drift from strict schemas; the fine‑tuned NER gives deterministic field labels.

---

### 3) Train the custom NER model
Code: `src/muril-train (1).ipynb`

Model: `google/muril-base-cased` fine‑tuned for token classification.

Key steps in the notebook:
- Loads the BIO dataset (created in step 1) and creates `label2id`/`id2label` maps.
- Tokenizes with alignment (`is_split_into_words=True`) and pads/truncates to fixed length.
- Adds domain tokens (e.g., locality names) to the tokenizer and resizes embeddings.
- Uses a custom `FocalLoss` with per‑class weights (computed from the training set, clipped for class balance).
- Trains with `TrainingArguments` and early stopping, evaluates every epoch, and persists the best checkpoint.

Artifacts saved:
- Final model and tokenizer (e.g., `muril_ner_logs/checkpoint-XXXX` or an explicit export folder like `muril_120k_5epochs`).

Notes:
- The notebook also includes an evaluation‑only path that computes a full classification report and confusion matrix.

---

### 4) Run inference and clustering
Code: `src/pipeline.ipynb`

Main flow:
1) Address standardization: `standardize_address_abbreviations_enhanced` expands/normalizes abbreviations (HN/H.No → house number, G No → gali number, etc.) and enforces spacing (alpha‑digit, digit‑alpha) for better tokenization.
2) Load the trained NER: creates a `transformers` pipeline (`aggregation_strategy="simple"`) to group sub‑tokens.
3) Batch inference: `batched_ner_out` runs NER in chunks for throughput; outputs a list of entity groups per address.
4) Post‑processing: `join_word` and `convert_single_output` stitch sub‑words back to clean strings and compose a `complete_address` from fields.
5) Embeddings and clustering (two paths):
   - Sentence‑level embeddings from the fine‑tuned MURIL encoder ([CLS]) for small demos.
   - Sentence‑Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) for large‑scale clustering.

FAISS indexing used:
- Locality‑scoped clustering uses cosine similarity via inner product on L2‑normalized embeddings:
  - Index: `faiss.IndexFlatIP` (inner product)
  - Embeddings normalized (`normalize_embeddings=True`) so IP ≡ cosine similarity.
- Demo neighbor search example also shows `IndexFlatL2` (exact L2).

Clustering algorithms used:
- Locality‑level neighbor grouping (simple threshold clustering):
  - Search top‑k neighbors in FAISS; if `cosine_sim >= 0.95`, assign to a cluster (visited set prevents duplicates). Produces `clusters.json` with `cluster_id`, `locality`, `canonical_address`, `customer_ids`.
- Hierarchical clustering (embeddings):
  - Computes cosine similarity → converts to distance → SciPy `linkage(..., method='ward')` and `fcluster` at a chosen distance threshold.
- Token‑level normalization (spell‑correction pass):
  - SBERT embeddings for tokens → `AgglomerativeClustering` with `distance_threshold=1.0`, `linkage='average'` → map each cluster to the most frequent representative.

Where to change thresholds:
- FAISS neighbor clustering: `similarity_threshold = 0.95`, `k = 10`.
- Hierarchical clustering: adjust dendrogram cut (`plt.axhline`) and `fcluster` distance.
- Token clustering: tune `distance_threshold` in `AgglomerativeClustering`.

Outputs:
- Parsed/normalized fields (`FIELDS + complete_address`) merged back onto source dataframes.
- Cluster JSON under `cluster_output/clusters.json`.
- Optional plots (dendrogram, confusion matrix) saved to disk.

---

### Appendix: LLM‑based JSON parsing (details and BIO bootstrapping)
Code: `src/multi_model_free_jsonator.py`

Purpose: Segment raw addresses into a fixed JSON schema using Groq LLMs with auto‑failover across multiple models and token/call budgets. Includes batching, JSON block extraction fallback, and CSV export with desired column order.

Use when:
- You want a non‑ML (rule/LLM) path to obtain structured address fields, either for bootstrapping training data or for comparison against the NER path.

LLM → BIO (bootstrapping custom NER training data):
1) Run `src/multi_model_free_jsonator.py` to produce a CSV with the schema columns:
   - `house_number, plot_number, floor, road_details, khasra_number, block, apartment, landmark, locality, area, locality2, village, pincode, complete_address`
2) Review/clean the LLM output (spot‑check fields and `complete_address`).
3) Feed that CSV to `src/bio_tagger.py` by setting it as `input_file` in `__main__`:
   - The tagger tokenizes `complete_address`, finds spans that match each field, and emits aligned BIO tags.
   - Output: a `sentence` + `tags` CSV ready for fine‑tuning in `muril-train (1).ipynb`.
4) Train the custom NER using the BIO dataset as described in section 2.

Notes:
- The tagger normalizes PIN codes to 6 digits and enforces consistent casing/spacing before span search.
- If you augment schema fields, add corresponding column names to `NERDatasetConverter.entity_columns` in `bio_tagger.py`.

---

### Quick start
1) Create training data: run `src/bio_tagger.py` on a structured CSV to produce a BIO‑tagged dataset.
2) Train NER: open `src/muril-train (1).ipynb`, point it to the BIO dataset, run through training, and save a final model folder (e.g., `muril_120k_5epochs`).
3) Inference + clustering: open `src/pipeline.ipynb`, update paths to your model and data, then run:
   - Standardize → NER batch → Post‑process → FAISS index → Cluster → Save outputs.

Tips:
- For cosine FAISS, always L2‑normalize embeddings and use `IndexFlatIP`.
- For large datasets, keep FAISS per‑locality to reduce search space and improve cluster quality.
- Keep `FIELDS` list consistent across parsing and export steps.

---

### Environment
- Python 3.10+
- Core libs: `transformers`, `datasets`, `torch`, `sentence-transformers`, `faiss`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `tqdm`.
- For LLM JSON parsing: Groq SDK and valid `GROQ_API_KEY` in environment.

Install (example):
```bash
pip install transformers datasets torch sentence-transformers faiss-cpu scikit-learn scipy pandas numpy tqdm
```

---

### File map (key parts)
- `src/bio_tagger.py`: from structured rows → BIO sentences/tags CSV (training data).
- `src/muril-train (1).ipynb`: trains the custom NER, evaluates, saves model/tokenizer.
- `src/pipeline.ipynb`: standardizes addresses, runs NER, post‑processes, builds FAISS indexes, and clusters (locality neighbor thresholding; hierarchical; token corrections).
- `src/multi_model_free_jsonator.py`: LLM auto‑failover JSON extraction and CSV export (optional path).


###Pipeline Stages
1.Data Collection

Input: 500K+ noisy, unstructured addresses (in Hinglish)
Output: Flat CSV file with ID, address columns

2. 🤖 Address Segmentation using LLMs

Used Groq's llama-3.1-8b-instant via LangChain
Prompt-based JSON extraction with fields: house_number, floor, block, locality, landmark, pincode, etc.
Batched and cached to handle Groq rate limits (30 calls/min, 15,000 tokens/min)
Output: Structured JSON per address, normalized to CSV

3. 🧬 BIO Tagging for NER Training

Converted Groq-segmented data into token-level BIO tags
Handled:
Pincode float bugs
Subword token alignment issues
Label overlap via priority order
Output: BIO-tagged CSV ready for NER fine-tuning

4. 🎓 NER Model Fine-Tuning

Model: google/muril-base-cased
Libraries: Transformers v4.53.1, PyTorch
Config:
Batch size: 80
Epochs: 10
Used DataCollatorWithPadding and TrainingConfig
Output: Fine-tuned NER model

5. 🔍 NER Inference & Structured Output

Used HuggingFace pipeline(aggregation_strategy="simple")
Extracted structured fields per address
Output: Normalized address field CSV

6. 🿃️‍♂️ Pre-Clustering: Manual Bucketing

Grouped addresses by PINCODE → LOCALITY → AREA
Reduced embedding computation scope

7. 🖐️ Embedding Generation

Used SBERT (paraphrase-MiniLM-L6-v2) to encode locality + house context
Normalized embeddings

8. 🧠 FAISS Clustering

Stage 1: FAISS IVF clustering at locality level (cosine similarity ~0.97)
Stage 2: Micro-clustering by house_number, floor, block using fuzzy match
Output: Clustered customer ID groups

9. ✅ Evaluation & Refinement

Metrics:
Cluster purity
Average size vs noise ratio
Manual spot-checks
Normalization: canonical house fields, synonym merging, token cleaning
Output: Final high-precision household clusters


###Discarded Approaches

-DistilBERT/mBERT for NER → weak results on Hinglish
-FastText, TF-IDF, or regex-only clustering → too shallow
-FlatIP for FAISS → RAM-hungry for 500K entries
-One-stage clustering → overclustered mess

###Learnings & Highlights

-Prompt-based annotation can bootstrap large NER datasets
-Locality-based pre-bucketing massively improves clustering performance
-Multi-stage clustering avoids semantic vs syntactic overlap errors
-Custom confidence scoring enables traceable cluster quality


###Future Enhancements:
•	Geocoding Integration: Map addresses to lat/lon coordinates via APIs (like Google Maps or OpenStreetMap) to enable geospatial validation.
•	Confidence-based Filtering: Use cluster confidence scores to allow business users to accept/reject edge cases.
•	NER Active Learning Loop: Feed back low-confidence model predictions into the fine-tuning pipeline for continuous improvement.
•	Multimodal Validation: Use delivery logs, billing history, or geo-tags to improve cluster validation.
•	Graph-Based Clustering: Build graphs using address entity overlaps and apply community detection (e.g., Louvain) for robust clustering.
•	LLM Finetuning: Continue fine-tuning domain-specific LLMs (e.g., on utility data, address complaints) to improve segmentation accuracy.
•	REST API Deployment: Turn the entire pipeline into an interactive microservice accessible by billing, delivery, and marketing teams.


