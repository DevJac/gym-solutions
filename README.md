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

4. Set the `JULIA_LOAD_PATH` environment variable to `src:`.

5. Run a script:

```fish
julia ppo_lander_discrete.jl
```

# Fix for Julia OpenGL issues

Replace Julia's `libstdc++.so.6` with a symlink to the system version:

* Delete julia-1.3.0/lib/julia/libstdc++.so.6
* Find the system libstdc++ with whereis libstdc++
* Link that location to julia-1.3.0/lib/julia-libstdc++.so.6 using ln.

See: https://github.com/JuliaPlots/Makie.jl/issues/614#issuecomment-635556398
See: https://github.com/JuliaGL/GLFW.jl/issues/198
