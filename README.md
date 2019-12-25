# Setup

1. Create a virtualenv:

```fish
python -m venv ~/.virtualenvs/gym
```

2. Activate the virtualenv:

```fish
source ~/.virtualenvs/gym/bin/activate.fish
```

3. Install Python packages:

```fish
pip install gym box2d-py matplotlib ipython
```

4. Start Julia REPL:

```fish
julia --project
```

5. In the Julia REPL, include and run desired scripts:

```julia
include("src/spinning_up/advantage_proximal_policy_optimization.jl")
run()
```
