# Needed changes to rl4co code

In `rl4co/rl4co/envs/routing/op/generator.py`, line 14:

```python
MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0, 500: 4.0, 1000: 4.0, 2000: 4.0, 5000: 4.0}
```

In `rl4co/rl4co/envs/routing/pctsp/generator.py`, line 12:

```python
MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0, 500: 9.0, 1000: 12.0, 2000: 15.0, 5000: 20.0}
```

