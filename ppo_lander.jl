using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere include("src/ppo_lander.jl")

policy = Policy()
train_until_reward!(policy, 200, fancy_output=true, save_policy=true)
