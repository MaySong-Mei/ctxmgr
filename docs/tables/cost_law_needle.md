# Peer-manager reads vs the trigger law T / (T - R) (Needle)

Built by `scripts/cost_law.py`. measured = manager prompt tokens / actor uncached prompt
tokens; predicted = mean context before an event / mean tokens the event frees (o200k).
ratio = measured / predicted. Means over episodes.
Across 55 cells, measured / predicted has median 0.94 (IQR 0.74-0.98).


| level | actor | condition | manager | version | n | events | measured | predicted | ratio |
|---|---|---|---|---|---|---|---|---|---|
| K2-C192 | 2B | peer_27b | 27B | edit-v8 | 8 | 45.4 | 1.14 | 1.20 | 0.95 |
| K2-C192 | 9B | peer_27b | 27B | edit-v8 | 8 | 43.8 | 1.11 | 1.15 | 0.97 |
| K4-C24 | 0.8B | peer_4b | 4B | edit-v8 | 8 | 14.0 | 3.52 | 6.14 | 0.57 |
| K4-C24 | 0.8B | peer_9b | 9B | edit-v8 | 8 | 4.2 | 1.12 | 1.09 | 1.02 |
| K4-C24 | 27B | peer_same | 27B | edit-v8 | 16 | 8.1 | 1.19 | 1.24 | 0.96 |
| K4-C24 | 2B | peer_27b | 27B | edit-v8 | 16 | 4.1 | 1.04 | 1.10 | 0.94 |
| K4-C24 | 2B | peer_27b | 27B | edit-v8+example | 8 | 4.0 | 0.84 | 1.06 | 0.80 |
| K4-C24 | 2B | peer_27b | 27B | edit-v8+mustnote | 8 | 4.0 | 0.83 | 1.09 | 0.76 |
| K4-C24 | 2B | peer_27b | 27B | edit-v8+para1 | 8 | 4.1 | 1.04 | 1.11 | 0.94 |
| K4-C24 | 2B | peer_27b | 27B | edit-v8+para2 | 8 | 4.0 | 1.03 | 1.10 | 0.94 |
| K4-C24 | 2B | peer_27b | 27B | edit-v8+para3 | 8 | 4.0 | 1.03 | 1.10 | 0.94 |
| K4-C24 | 2B | peer_27b | 27B | edit-v8+para4 | 8 | 4.0 | 1.04 | 1.07 | 0.98 |
| K4-C24 | 2B | peer_27b | 27B | edit-v8+think | 8 | 4.8 | 1.17 | 1.23 | 0.95 |
| K4-C24 | 2B | peer_4b | 4B | edit-v8 | 8 | 10.9 | 3.61 | 8.35 | 0.43 |
| K4-C24 | 2B | peer_9b | 9B | edit-v8 | 7 | 6.4 | 1.81 | 1.58 | 1.15 |
| K4-C24 | 2B | peer_9b | 9B | edit-v8+example | 8 | 4.8 | 1.02 | 1.38 | 0.74 |
| K4-C24 | 2B | peer_9b | 9B | edit-v8+mustnote | 8 | 5.8 | 1.03 | 1.37 | 0.75 |
| K4-C24 | 2B | peer_9b | 9B | edit-v8+para2 | 4 | 8.5 | 2.32 | 2.04 | 1.14 |
| K4-C24 | 2B | peer_9b | 9B | edit-v8+para3 | 6 | 5.5 | 1.54 | 1.36 | 1.14 |
| K4-C24 | 2B | peer_9b | 9B | edit-v8+para4 | 6 | 6.5 | 1.85 | 1.60 | 1.15 |
| K4-C24 | 2B | peer_9b | 9B | edit-v8+think | 7 | 6.3 | 2.16 | 2.76 | 0.78 |
| K4-C24 | 2B | peer_same | 2B | edit-v8 | 7 | 4.0 | 3.07 | 24.41 | 0.13 |
| K4-C24 | 4B | peer_27b | 27B | edit-v8 | 16 | 4.2 | 1.07 | 1.09 | 0.98 |
| K4-C24 | 4B | peer_27b | 27B | edit-v8+think | 8 | 6.2 | 1.44 | 1.27 | 1.14 |
| K4-C24 | 4B | peer_9b | 9B | edit-v8 | 8 | 4.1 | 1.08 | 1.06 | 1.02 |
| K4-C24 | 4B | peer_9b | 9B | edit-v8+example | 8 | 4.1 | 0.86 | 1.07 | 0.80 |
| K4-C24 | 4B | peer_9b | 9B | edit-v8+mustnote | 8 | 4.0 | 0.83 | 1.03 | 0.81 |
| K4-C24 | 4B | peer_9b | 9B | edit-v8+think | 8 | 5.2 | 2.12 | 60.54 | 0.04 |
| K4-C24 | 4B | peer_same | 4B | edit-v8 | 8 | 9.2 | 7.26 | 194.95 | 0.04 |
| K4-C24 | 9B | peer_27b | 27B | edit-v8 | 17 | 4.2 | 1.06 | 1.09 | 0.97 |
| K8-C48 | 0.8B | peer_4b | 4B | edit-v8 | 8 | 11.2 | 3.63 | 8.65 | 0.42 |
| K8-C48 | 0.8B | peer_9b | 9B | edit-v8 | 8 | 10.0 | 1.10 | 1.14 | 0.96 |
| K8-C48 | 2B | peer_27b | 27B | edit-v8+example | 8 | 9.2 | 0.97 | 1.08 | 0.89 |
| K8-C48 | 2B | peer_27b | 27B | edit-v8+mustnote | 8 | 11.8 | 1.22 | 1.44 | 0.85 |
| K8-C48 | 2B | peer_27b | 27B | edit-v8+para1 | 8 | 11.2 | 1.14 | 1.20 | 0.95 |
| K8-C48 | 2B | peer_27b | 27B | edit-v8+para2 | 8 | 10.2 | 1.53 | 1.79 | 0.86 |
| K8-C48 | 2B | peer_27b | 27B | edit-v8+para3 | 8 | 5.0 | 2.24 | 2.86 | 0.78 |
| K8-C48 | 2B | peer_27b | 27B | edit-v8+para4 | 8 | 9.1 | 1.06 | 1.09 | 0.97 |
| K8-C48 | 2B | peer_27b | 27B | edit-v8+think | 8 | 11.6 | 2.12 | 3.12 | 0.68 |
| K8-C48 | 2B | peer_4b | 4B | edit-v8 | 8 | 6.1 | 2.87 | 9.16 | 0.31 |
| K8-C48 | 2B | peer_9b | 9B | edit-v8 | 8 | 9.9 | 1.14 | 1.12 | 1.01 |
| K8-C48 | 2B | peer_9b | 9B | edit-v8+example | 8 | 7.5 | 1.40 | 2.09 | 0.67 |
| K8-C48 | 2B | peer_9b | 9B | edit-v8+mustnote | 8 | 10.0 | 1.02 | 1.37 | 0.74 |
| K8-C48 | 2B | peer_9b | 9B | edit-v8+para1 | 3 | 10.3 | 1.97 | 2.06 | 0.96 |
| K8-C48 | 2B | peer_9b | 9B | edit-v8+para2 | 8 | 11.4 | 1.31 | 1.27 | 1.03 |
| K8-C48 | 2B | peer_9b | 9B | edit-v8+para3 | 8 | 9.5 | 1.07 | 1.07 | 0.99 |
| K8-C48 | 2B | peer_9b | 9B | edit-v8+para4 | 8 | 10.4 | 1.11 | 1.16 | 0.96 |
| K8-C48 | 2B | peer_9b | 9B | edit-v8+think | 6 | 11.2 | 2.56 | 3.52 | 0.73 |
| K8-C48 | 2B | peer_same | 2B | edit-v8 | 2 | 3.0 | 2.07 | 17.56 | 0.12 |
| K8-C48 | 4B | peer_27b | 27B | edit-v8+think | 8 | 16.0 | 1.78 | 1.82 | 0.98 |
| K8-C48 | 4B | peer_9b | 9B | edit-v8 | 8 | 9.1 | 1.08 | 1.04 | 1.04 |
| K8-C48 | 4B | peer_9b | 9B | edit-v8+example | 8 | 9.6 | 0.98 | 1.11 | 0.88 |
| K8-C48 | 4B | peer_9b | 9B | edit-v8+mustnote | 8 | 9.1 | 0.97 | 1.04 | 0.93 |
| K8-C48 | 4B | peer_9b | 9B | edit-v8+think | 8 | 9.2 | 5.81 | 47.35 | 0.12 |
| K8-C48 | 4B | peer_same | 4B | edit-v8 | 8 | 9.8 | 5.30 | 10.83 | 0.49 |
