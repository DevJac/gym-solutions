using OpenAIGym
env = GymEnv(:LunarLander, :v2)
for i ∈ 1:5
    T = 0
    R = run_episode(env, RandomPolicy()) do (s, a, r, s′)
        render(env)
        T += 1
    end
    @info("Episode $i finished after $T steps", R)
end
close(env)
