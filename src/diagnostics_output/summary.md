# HERMES — Error Diagnostics (Full Pipeline (4 SECs))

**File:** `results_current.json`  
**Format:** Full Pipeline (4 SECs)  
**Classes:** positive, negative, neutral  
**Total evaluated:** 5 (skipped: 0)  
**Correct:** 2  
**Errors:** 3  

## Metrics

| Metric | Value |
|--------|-------|
| AvgRec | 0.2222 |
| Macro-F1 | 0.1905 |
| F1^PN | 0.2857 |
| Accuracy | 0.4000 |
| Micro-F1 | 0.4000 |

## Per-Class Metrics

| Class | Precision | Recall | F1 | Gold | Pred |
|-------|-----------|--------|----|------|------|
| positive | 0.5000 | 0.6667 | 0.5714 | 3 | 4 |
| negative | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| neutral | 0.0000 | 0.0000 | 0.0000 | 2 | 1 |

## Confusion Matrix

| Gold \ Pred | Positive | Negative | Neutral |
|------------|-------:|-------:|-------:|
| positive | 2 | 0 | 1 |
| negative | 0 | 0 | 0 |
| neutral | 2 | 0 | 0 |

## Error Distribution

| Pattern | Count | % of Total Errors | File |
|---------|-------|------------------:|------|
| neutral>positive | 2 | 66.7% | `errors_neutral_to_positive.md` |
| positive>neutral | 1 | 33.3% | `errors_positive_to_neutral.md` |

## Resource Usage

| Metric | Total | Avg per Tweet |
|--------|------:|--------------:|
| Input tokens | 42,817 | 8,563 |
| Output tokens | 9,513 | 1,903 |
| Total tokens | 52,330 | 10,466 |
| Time (seconds) | 64.7 | 12.9 |
