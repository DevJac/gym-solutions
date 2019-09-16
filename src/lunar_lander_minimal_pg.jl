# Based on: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
using Flux
using OpenAIGym
using Plots
using Printf

pyplot()

function OpenAIGym.sample(set::DiscreteSet, weights)
    sample(set.items, weights)
end

const env = GymEnv(:LunarLander, :v2)

function make_policy_network(hidden_layer_size=32)
    Chain(
        Dense(length(env.state), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, length(env.actions), identity),
        softmax)
end

function policy_network_loss(policy, sars)
    -sum(sars) do sars
        # The training process will seek to maximize the following formula.
        # The policy network outputs a probability distribution over all actions.
        # The log of a probability is a negative number. Smaller probabilities are exponentially more negative.
        # Thus, to maximize the formula, the policy should output high probabilities for actions with high rewards.
        log(policy.policy_network(sars.s)[sars.a + 1]) * sars.q
    end / length(filter(sars -> sars.f, sars))
end

const policy_network_optimizer = AMSGrad()

function train_policy!(policy, sars)
    Flux.train!(sars -> policy_network_loss(policy, sars), Flux.params(policy.policy_network), [(sars,)], policy_network_optimizer)
end

struct Policy <: AbstractPolicy
    policy_network
end

function Reinforce.action(policy::Policy, r, s, A)
    sample(env.actions, Weights(policy.policy_network(s)))
end

mutable struct SARS
    s
    a
    r
    q
    s_next
    f
end

function run_episodes(n_episodes, policy; render_all=false, render_one=false, close_env=false)
    sars = []
    rewards = []
    for episode in 1:n_episodes
        reward = run_episode(env, policy) do (s, a, r, s_next)
            push!(sars, SARS(s, a, r, nothing, s_next, finished(env)))
            if render_all; render(env) end
            if !render_all && render_one && episode == 1; render(env) end
        end
        push!(rewards, reward)
    end
    if close_env; close(env) end
    fill_q!(sars)
    sars, rewards
end

function fill_q!(sars; discount_factor=1)
    q = 0
    for i in length(sars):-1:1
        q *= discount_factor
        if sars[i].f
            q = 0
        end
        q += sars[i].r
        sars[i].q = q
    end
end

last(n, xs) = xs[max(1, end-n+1):end]

function run(policy=Policy(make_policy_network()), stop_reward=200)
    try
        all_rewards = []
        batch_size = 1
        best_mean_reward = -Inf
        for iteration in Iterators.countfrom(1)
            @printf("Batch Size: %3d ", batch_size)
            sars, rewards = run_episodes(batch_size, policy)
            append!(all_rewards, rewards)
            recent_rewards = last(100, all_rewards)
            if mean(recent_rewards) > best_mean_reward && length(recent_rewards) == 100
                best_mean_reward = mean(recent_rewards)
                batch_size = max(1, floor(batch_size * 0.9))
            else
                batch_size += 1
            end
            @printf("Episodes: %5d Recent Rewards: %7.2f\n", length(all_rewards), mean(recent_rewards))
            display(scatter(last(10_000, all_rewards), size=(1200, 800), legend=false, markeralpha=0.8, markersize=3, markerstrokewidth=0))
            if mean(recent_rewards) >= stop_reward; break end
            train_policy!(policy, sars)
        end
        @printf("Solved in %d episodes. Observe the solution.", length(all_rewards))
        run_episodes(1000, policy, render_all=true)
    finally
        close(env)
    end
    policy
end
