# Qwen3 Optimizer Comparison - Evaluation Summary

**Total Models Evaluated:** 5
**Evaluation Date:** 2025-11-20 18:43:57

## Best Performing Model
- **Model:** adabound
- **Accuracy:** 0.2097
- **Speed:** 4.63 items/sec

## Results Summary

| Model | Accuracy | Speed (items/s) | Eval Time (s) | GPU Memory (MB) |
|-------|----------|-----------------|---------------|-----------------|
| adabound | 0.2097 | 4.63 | 263.7 | 0 |
| hybrid | 0.2072 | 4.66 | 262.2 | 0 |
| adamw | 0.2056 | 4.64 | 263.3 | 0 |
| sgd | 0.2015 | 4.64 | 263.2 | 0 |
| baseline | 0.1957 | 4.62 | 264.5 | 0 |

## Performance Analysis
### Improvement over Baseline
- **adabound:** +1.39% accuracy improvement
- **hybrid:** +1.15% accuracy improvement
- **adamw:** +0.98% accuracy improvement
- **sgd:** +0.57% accuracy improvement
### Speed
- **Fastest:** hybrid (4.66 items/s)
- **Slowest:** baseline (4.62 items/s)
- **Speedup:** 1.01x