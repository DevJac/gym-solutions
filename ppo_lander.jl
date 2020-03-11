using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

using Runner

@everywhere include("src/ppo_lander.jl")

policy = Policy()
batch_train_until_reward!(env, policy, 200, fancy_output=true, save_policy=true)
