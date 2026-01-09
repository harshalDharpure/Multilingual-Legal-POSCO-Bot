# Comprehensive Research Plan: Multilingual Legal Dialogue System for Access-to-Justice

## 1. Dataset Analysis & Stratified Splitting Strategy

### 1.1 Current Dataset Statistics
- **Total Samples:** 1200 (400 per language)
- **Languages:** Hindi, English, Code-mixed
- **Complexity Distribution:**
  - Layman: 133 per language (399 total)
  - Intermediate: 133 per language (399 total)
  - Professional: 134 per language (402 total)
- **Turn Distribution (Buckets):**
  - Bucket A: 2-3 turns
  - Bucket B: 3-4 turns
  - Bucket C: 4-5 turns
  - Bucket D: 5-6 turns

### 1.2 Stratified Train/Validation/Test Split (70/15/15)

**Key Principle:** Preserve proportional distribution across:
1. Language (33.33% each)
2. Complexity level (Layman/Intermediate/Professional)
3. Bucket distribution (A/B/C/D)
4. Turn count distribution

**Split Strategy:**
```
Total: 1200 samples
├── Train: 840 samples (70%)
│   ├── Hindi: 280 samples (23.33%)
│   ├── English: 280 samples (23.33%)
│   └── Code-mixed: 280 samples (23.33%)
├── Validation: 180 samples (15%)
│   ├── Hindi: 60 samples (5%)
│   ├── English: 60 samples (5%)
│   └── Code-mixed: 60 samples (5%)
└── Test: 180 samples (15%)
    ├── Hindi: 60 samples (5%)
    ├── English: 60 samples (5%)
    └── Code-mixed: 60 samples (5%)
```

**Within each language split, preserve:**
- **Train (per language):** 
  - Layman: 93 samples (31%)
  - Intermediate: 93 samples (31%)
  - Professional: 94 samples (31.33%)
- **Val/Test (per language):**
  - Layman: 20 samples each
  - Intermediate: 20 samples each
  - Professional: 20 samples each

**Bucket preservation:** Ensure each split maintains original bucket proportions within complexity levels.

### 1.3 Implementation Algorithm (Stratified Split)
```python
# Pseudo-code for stratified split
def stratified_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Multi-level stratification:
    1. Language
    2. Complexity (layman/intermediate/professional)
    3. Bucket (A/B/C/D)
    4. Random seed for reproducibility
    """
    splits = {'train': [], 'val': [], 'test': []}
    
    for language in ['hindi', 'english', 'code_mixed']:
        for complexity in ['layman', 'intermediate', 'professional']:
            for bucket in ['A', 'B', 'C', 'D']:
                subset = filter_by_strata(dataset, language, complexity, bucket)
                # Split this subset maintaining ratios
                sub_train, sub_val, sub_test = split_subset(subset, train_ratio, val_ratio, test_ratio)
                splits['train'].extend(sub_train)
                splits['val'].extend(sub_val)
                splits['test'].extend(sub_test)
    
    # Shuffle each split
    return splits
```

---

## 2. Model Selection for Fine-tuning

### 2.1 Multilingual Base Models (Recommended Priority Order)

#### **Tier 1: State-of-the-Art Multilingual LLMs**
1. **LLaMA-3.1 (70B/8B) + Multilingual Extension**
   - Rationale: Strong multilingual capabilities, legal domain adaptability
   - Parameter count: 8B (efficient) or 70B (SOTA)
   - Languages: Excellent Hindi/English support
   
2. **Mistral-7B-Instruct-v0.3**
   - Rationale: Strong instruction following, multilingual support
   - Fine-tuning: QLoRA for efficient training
   
3. **Qwen2.5 (7B/14B) - Multilingual**
   - Rationale: Strong performance on multilingual benchmarks
   - Code-mixed handling: Better than most models
   
4. **Phi-3.5 (3.8B) - Multilingual**
   - Rationale: Efficient, good instruction following
   - Training cost: Lower compute requirements

#### **Tier 2: Legal-Domain Specialized Models**
5. **LegalBERT (Legal domain pretrained)**
   - Approach: Fine-tune for dialogue generation
   - Advantage: Legal vocabulary understanding
   
6. **Legal-Llama (if available)**
   - Domain: Legal-specific pretraining
   - Multilingual: May need extension

#### **Tier 3: Indian Language Focused**
7. **Airavata (Hindi-focused)**
   - Specialization: Hindi language models
   - Extension: Add English and code-mixed support
   
8. **MuRIL (Multilingual Representations for Indian Languages)**
   - Base: BERT-based for Indian languages
   - Adaptation: Convert to generative model

### 2.2 Fine-tuning Strategy

**Approach 1: Full Fine-tuning (for smaller models < 7B)**
- Learning rate: 2e-5 to 5e-5
- Batch size: 4-8 (depending on GPU)
- Epochs: 3-5 with early stopping
- Warmup: 10% of training steps

**Approach 2: Parameter-Efficient Fine-tuning (PEFT) - Recommended**
- **LoRA (Low-Rank Adaptation)**
  - Rank: 16-32
  - Alpha: 32-64
  - Target modules: q_proj, v_proj, k_proj, o_proj
  - Advantages: Faster, lower memory, prevents catastrophic forgetting
  
- **QLoRA (Quantized LoRA)**
  - 4-bit quantization
  - Further memory reduction
  - Best for: Models > 13B parameters

**Approach 3: Adapter Layers**
- Lightweight adaptation
- Preserves base model knowledge
- Easy multi-task extension

---

## 3. Evaluation Framework

### 3.1 Few-Shot and Zero-Shot Evaluation

**Zero-Shot Setup:**
- Base model (no fine-tuning) tested on test set
- Measures: Out-of-the-box performance
- Baseline for comparison

**Few-Shot Setup:**
- K-shot examples: K ∈ {1, 3, 5, 10}
- Selection strategy: 
  - Stratified sampling by complexity and language
  - Diversity: Ensure different buckets represented
- Prompt format:
  ```
  System: You are a legal assistant helping with POCSO cases...
  
  Example 1:
  User: [example 1 user turn]
  Assistant: [example 1 assistant turn]
  
  ...
  
  Example K:
  User: [example K user turn]
  Assistant: [example K assistant turn]
  
  Current Query:
  User: [test query]
  Assistant:
  ```

### 3.2 Evaluation Metrics

#### **A. Automatic Metrics**

1. **BLEU Score (n-gram overlap)**
   - BLEU-1, BLEU-2, BLEU-3, BLEU-4
   - Captures: Lexical similarity
   - Limitation: May penalize semantic paraphrases

2. **ROUGE Scores**
   - ROUGE-L (Longest Common Subsequence)
   - ROUGE-1, ROUGE-2 (n-gram recall)
   - Captures: Recall-oriented evaluation

3. **METEOR**
   - Alignment-based metric
   - Handles: Synonyms and paraphrases better than BLEU
   - Good for: Legal domain with varied terminology

4. **BERTScore**
   - Semantic similarity using BERT embeddings
   - Multilingual: Use multilingual BERT
   - Advantages: Captures semantic meaning, not just n-grams
   - **Recommended:** Primary automatic metric

5. **MoverScore**
   - Word mover's distance in embedding space
   - Good for: Long-form responses

6. **BLEURT (Learning-based)**
   - Trained on human judgments
   - Better correlation with human evaluation
   - Use: Multilingual BLEURT if available

7. **Perplexity**
   - Language modeling metric
   - Lower is better
   - Measures: Model confidence

#### **B. Legal Domain-Specific Metrics**

8. **Statute Citation Accuracy**
   - Extract: POCSO sections, IPC sections, CrPC sections
   - Compare: Ground truth vs. predicted citations
   - Precision, Recall, F1 for citations
   - **Novel Contribution:** Automated citation validation

9. **Legal Term Precision**
   - Legal vocabulary usage correctness
   - Term extraction: Use legal NER models
   - Validate: Against legal terminology dictionaries

10. **Section Number Hallucination Rate**
    - Check: If cited sections exist in actual statutes
    - Validation: Against official POCSO/IPC/CrPC text
    - **Novel Metric:** Legal fact-checking

11. **Advice Safety Score**
    - Binary: Safe vs. Unsafe legal advice
    - Classifier: Train on legal advice safety guidelines
    - Critical: No false legal guarantees

#### **C. Dialogue Quality Metrics**

12. **Turn Coherence**
    - Measures: Response relevance to user query
    - Use: Cross-encoder models for relevance scoring

13. **Empathy Score**
    - Evaluate: Trauma-informed language usage
    - Method: Fine-tune sentiment/empathy classifier
    - **Novel Contribution:** Legal domain empathy evaluation

14. **Comprehensibility Score**
    - For layman queries: Response should be simple
    - For professional queries: Technical accuracy required
    - Adapt: Complexity-appropriate evaluation

#### **D. Multilingual Metrics**

15. **Code-Mixing Quality**
    - Measure: Natural Hindi-English mixing
    - Use: Code-mixed language models (MuRIL, HingBERT)
    - **Novel Contribution:** Code-mixed evaluation in legal domain

16. **Language Consistency**
    - Check: Response language matches query language
    - Exception: Code-mixed queries → code-mixed responses

---

## 4. Hallucination Detection Framework

### 4.1 Multi-Dimensional Hallucination Detection

**Definition:** A hallucination occurs when the model generates:
1. **Factual Errors:** Incorrect legal facts, non-existent statutes
2. **Citation Errors:** Wrong section numbers, non-existent sections
3. **Contradictory Information:** Self-contradicting statements
4. **Fabricated Cases:** Made-up case names, judgments
5. **Incorrect Procedures:** Wrong legal procedures, timelines

### 4.2 Implementation Strategy

#### **Level 1: Citation Validation (High Precision)**
```python
def validate_statute_citations(response, ground_truth_statutes):
    """
    Extract all statute citations from response:
    - POCSO Section X
    - IPC Section Y
    - CrPC Section Z
    
    Validate:
    1. Section number exists in actual statute
    2. Section is relevant to the case context
    3. Section interpretation is correct
    """
    extracted_citations = extract_citations(response)
    hallucinated_citations = []
    
    for citation in extracted_citations:
        if not exists_in_statute(citation):
            hallucinated_citations.append(citation)
        elif not is_relevant(citation, context):
            hallucinated_citations.append(citation)
    
    hallucination_rate = len(hallucinated_citations) / len(extracted_citations)
    return hallucination_rate, hallucinated_citations
```

#### **Level 2: Fact Verification (Medium Precision)**
- **Legal Knowledge Base:**
  - POCSO Act full text (Section 1-46)
  - IPC relevant sections (354, 354A, 376, 377, 506)
  - CrPC relevant sections (154, 161, 164, 173)
  - JJ Act relevant sections
  
- **Validation Methods:**
  1. **Retrieval-Augmented Verification:**
     - Retrieve relevant legal text for each claim
     - Compare: Generated response vs. retrieved text
     - Use: Dense retrieval (DPR) or sparse retrieval (BM25)
     
  2. **Entailment Checking:**
     - Use: Legal NLI models (if available) or general NLI
     - Check: Does legal KB entail the claim?
     - Models: RoBERTa-Large-MNLI (multilingual variant)

#### **Level 3: Semantic Consistency (Lower Precision, Higher Recall)**
- **Self-Consistency:**
  - Generate multiple responses (temperature sampling)
  - Check: Consistency across responses
  - Low consistency → potential hallucination
  
- **Cross-Validation with Ground Truth:**
  - Compare: Generated response vs. reference response
  - Focus: On factual claims, not wording
  - Use: BERTScore for semantic similarity

#### **Level 4: Legal Procedure Validation**
```python
def validate_legal_procedures(response, case_context):
    """
    Check if described procedures are correct:
    - FIR filing process
    - Statement recording (161, 164 CrPC)
    - Medical examination procedures
    - Court procedures
    - Timeline accuracy
    """
    procedures = extract_procedures(response)
    validated = []
    
    for proc in procedures:
        if not is_valid_procedure(proc, case_context):
            validated.append(('hallucinated', proc))
        else:
            validated.append(('correct', proc))
    
    return validated
```

### 4.3 Hallucination Rate Calculation

**Formula (Inspired by medical LLM evaluation papers):**
```
Hallucination Rate = (Number of Hallucinated Claims) / (Total Number of Claims)

Where a "claim" is:
1. A statute citation (Section X of Act Y)
2. A legal procedure statement
3. A factual legal assertion
4. A case reference (if mentioned)
```

**Aggregated Metrics:**
1. **Citation Hallucination Rate:** Citations that don't exist or are incorrect
2. **Factual Hallucination Rate:** Legal facts that are incorrect
3. **Procedural Hallucination Rate:** Procedures that are wrong
4. **Overall Hallucination Rate:** Weighted average

**Weighted Formula:**
```
Overall HR = 0.4 × Citation_HR + 0.4 × Factual_HR + 0.2 × Procedural_HR
```

**Per-Complexity Analysis:**
- Layman queries: Higher tolerance (simplification may introduce minor inaccuracies)
- Professional queries: Zero tolerance (must be legally precise)

---

## 5. Ground Truth Validation

### 5.1 Reference Response Quality Check

**Validation against `all_case_passages.txt`:**

1. **Case Alignment:**
   - Map each dialogue to source case
   - Verify: Dialogue reflects case facts accurately
   - Check: No contradictions with case details

2. **Legal Accuracy:**
   - Cross-reference: Response with case passages
   - Validate: Statute citations mentioned in case
   - Verify: Procedures align with case description

3. **Completeness Check:**
   - Ensure: All key legal aspects covered
   - Verify: No missing critical information

### 5.2 Expert Validation Protocol

**Phase 1: Legal Expert Review (Recommended)**
- Sample: 10-20% of test set (18-36 samples)
- Reviewers: 2-3 legal experts (POCSO domain)
- Criteria:
  - Legal accuracy
  - Advice safety
  - Completeness
  - Empathy and trauma-informed approach

**Phase 2: Inter-Annotator Agreement**
- Cohen's Kappa for categorical judgments
- Krippendorff's Alpha for continuous scores
- Target: κ > 0.7, α > 0.8

---

## 6. Novel Contributions & Research Innovation

### 6.1 Novel Contributions

#### **Contribution 1: Multilingual Legal Dialogue System**
- **Novelty:** First comprehensive Hindi-English-Code-mixed legal dialogue system
- **Impact:** Addresses real-world need for multilingual access-to-justice
- **Publication Angle:** ACL, EMNLP, LREC (multilingual NLP conferences)

#### **Contribution 2: Complexity-Adaptive Legal Assistance**
- **Novelty:** System adapts response complexity to user's legal literacy level
- **Technical:** Complexity-aware fine-tuning and evaluation
- **Impact:** Better accessibility for layman users

#### **Contribution 3: Legal Hallucination Detection Framework**
- **Novelty:** Multi-dimensional hallucination detection for legal domain
- **Components:**
  - Citation validation
  - Procedure verification
  - Fact-checking pipeline
- **Publication Angle:** LegalTech, AI & Law conferences

#### **Contribution 4: Code-Mixed Legal NLP**
- **Novelty:** First work on code-mixed legal dialogue
- **Challenge:** Natural Hinglish in legal context
- **Evaluation:** Novel metrics for code-mixed quality

#### **Contribution 5: Trauma-Informed Legal AI**
- **Novelty:** Evaluation of empathy and trauma-informed language
- **Impact:** Ethical AI in sensitive domains
- **Publication Angle:** FAccT, AIES (AI ethics conferences)

#### **Contribution 6: Few-Shot Legal Domain Adaptation**
- **Novelty:** Systematic evaluation of few-shot learning for legal dialogue
- **Analysis:** How many examples needed for good performance?
- **Practical Impact:** Lower data requirements

### 6.2 Research Questions

1. **RQ1:** Can multilingual LLMs effectively handle code-mixed legal queries?
2. **RQ2:** How does fine-tuning compare to few-shot prompting for legal dialogue?
3. **RQ3:** What is the hallucination rate in legal domain LLM responses?
4. **RQ4:** Can complexity-adaptive responses improve user comprehension?
5. **RQ5:** How do different fine-tuning methods (LoRA vs. full) affect legal accuracy?

### 6.3 Experimental Design

**Experiment 1: Baseline Comparison**
- Zero-shot vs. Few-shot (K=1,3,5,10) vs. Fine-tuned
- Models: LLaMA-3.1, Mistral-7B, Qwen2.5
- Metrics: BERTScore, Citation Accuracy, Hallucination Rate

**Experiment 2: Fine-tuning Ablation**
- Full fine-tuning vs. LoRA vs. QLoRA
- Parameters: Rank, alpha, target modules
- Analysis: Performance vs. compute trade-off

**Experiment 3: Multilingual Analysis**
- Per-language breakdown
- Cross-lingual transfer (train on Hindi, test on English)
- Code-mixed handling quality

**Experiment 4: Complexity-Aware Evaluation**
- Separate metrics for layman vs. professional queries
- Complexity-adaptive fine-tuning
- User comprehension study (optional)

**Experiment 5: Hallucination Analysis**
- Error categorization
- Most common hallucination types
- Mitigation strategies

---

## 7. Research Methodology Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATASET PREPARATION                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Hindi      │  │   English    │  │  Code-mixed  │         │
│  │  (400)       │  │   (400)      │  │   (400)      │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│           │                │                │                  │
│           └────────────────┼────────────────┘                  │
│                            │                                   │
│                    ┌───────▼────────┐                          │
│                    │ Stratified     │                          │
│                    │ Split          │                          │
│                    └───────┬────────┘                          │
│                            │                                   │
│         ┌──────────────────┼──────────────────┐               │
│         │                  │                  │                │
│    ┌────▼────┐      ┌──────▼──────┐    ┌─────▼─────┐         │
│    │  Train  │      │ Validation  │    │   Test    │         │
│    │  (840)  │      │   (180)     │    │   (180)   │         │
│    └─────────┘      └─────────────┘    └───────────┘         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Base Models: LLaMA-3.1, Mistral-7B, Qwen2.5           │  │
│  └─────────────────────────────────────────────────────────┘  │
│                            │                                   │
│         ┌──────────────────┼──────────────────┐               │
│         │                  │                  │                │
│    ┌────▼────┐      ┌──────▼──────┐    ┌─────▼─────┐         │
│    │  Zero-  │      │  Few-Shot   │    │  Fine-    │         │
│    │  Shot   │      │  (K=1,3,5)  │    │  tuned    │         │
│    │         │      │             │    │ (LoRA/    │         │
│    │         │      │             │    │  Full)    │         │
│    └─────────┘      └─────────────┘    └───────────┘         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION PIPELINE                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    Test Set (180)                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                            │                                   │
│         ┌──────────────────┼──────────────────┐               │
│         │                  │                  │                │
│    ┌────▼──────────┐  ┌────▼───────────┐  ┌──▼────────────┐  │
│    │  Automatic    │  │  Legal-Specific│  │  Hallucination│  │
│    │  Metrics      │  │  Metrics       │  │  Detection    │  │
│    │               │  │                │  │               │  │
│    │ • BERTScore   │  │ • Citation     │  │ • Citation    │  │
│    │ • BLEU        │  │   Accuracy     │  │   Validation  │  │
│    │ • ROUGE       │  │ • Legal Term   │  │ • Fact        │  │
│    │ • METEOR      │  │   Precision    │  │   Verification│  │
│    │               │  │ • Advice       │  │ • Procedure   │  │
│    │               │  │   Safety       │  │   Validation  │  │
│    └───────────────┘  └────────────────┘  └───────────────┘  │
│                            │                                   │
│                            ▼                                   │
│         ┌───────────────────────────────────────────┐          │
│         │     Ground Truth Validation               │          │
│         │  (all_case_passages.txt + Expert Review) │          │
│         └───────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSIS & INSIGHTS                          │
│                                                                 │
│  • Per-language performance                                     │
│  • Per-complexity analysis                                      │
│  • Hallucination rate breakdown                                 │
│  • Error categorization                                         │
│  • Few-shot vs. fine-tuning comparison                          │
│  • Complexity-adaptive effectiveness                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementation Roadmap

### Phase 1: Dataset Preparation (Week 1-2)
- [ ] Implement stratified split algorithm
- [ ] Validate split proportions
- [ ] Create train/val/test JSONL files
- [ ] Verify no data leakage

### Phase 2: Baseline Experiments (Week 3-4)
- [ ] Zero-shot evaluation on all models
- [ ] Few-shot evaluation (K=1,3,5,10)
- [ ] Baseline metrics calculation

### Phase 3: Fine-tuning (Week 5-8)
- [ ] Setup training infrastructure (HuggingFace, PEFT)
- [ ] Hyperparameter tuning on validation set
- [ ] Fine-tune top 3 models (LLaMA, Mistral, Qwen)
- [ ] Evaluate on validation set

### Phase 4: Comprehensive Evaluation (Week 9-10)
- [ ] Test set evaluation (all models)
- [ ] Implement hallucination detection pipeline
- [ ] Citation validation system
- [ ] Ground truth alignment

### Phase 5: Analysis & Writing (Week 11-14)
- [ ] Statistical analysis
- [ ] Error analysis
- [ ] Paper writing
- [ ] Figure generation

---


