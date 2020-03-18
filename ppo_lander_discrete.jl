using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

using Runner

@everywhere include("src/ppo_lander_discrete.jl")

policy = Policy()
batch_train_until_reward!(policy, 200, batch_size=200, fancy_output=true, save_policy=true)
