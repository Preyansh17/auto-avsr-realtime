| set | n | validity | readability (<=0.372) | pooled WER | WER-0 | sharp (>=0.7) | likeness range/speed/spw | in band |
|---|---|---|---|---|---|---|---|---|
| a | 30 | 1.00 | 0.142 | 12.4% | 0.63 | 0.52 | 0.313/1.611/0.347 | False |
| b | 100 | 1.00 | 0.190 | 18.4% | 0.43 | 0.52 | 0.288/1.604/0.360 | False |
| d | 169 | 1.00 | 0.142 | 13.2% | 0.53 | 0.50 | 0.303/1.678/0.340 | False |
| real_a | 30 | 1.00 | 0.231 | 23.3% | 0.30 | 1.02 | 0.401/2.405/0.398 | True |

Real band p10..p90: range 0.308..0.545, speed 1.835..3.399, spw 0.296..0.587

Checks: {'a_read': True, 'b_read': True, 'd_read': True, 'b_valid': True, 'sharp': False, 'likeness': False}

G4: **FAIL**; stop rule (a > 0.6) fires: False
