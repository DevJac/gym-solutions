using Flux
using OpenAIGym
using Plots
using Printf

pyplot()

Base.iterate(set::DiscreteSet) = iterate(set.items)
Base.iterate(set::DiscreteSet, state) = iterate(set.items, state)
Flux.onehot(a, set::DiscreteSet) = Flux.onehot(a, set.items)
OpenAIGym.sample(set::DiscreteSet, weights) = sample(set.items, weights)

const env = GymEnv(:LunarLander, :v2)

function make_π_network(hidden_layer_size=32)
    Chain(
        Dense(length(env.state), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, length(env.actions), identity),
        softmax)
end

a_to_π_index(a) = indexin(a, env.actions.items)[1]

clip(x, lo, hi) = clamp(x, Float32(lo), Float32(hi))

function π_loss(policy₀, policy′, sars, ϵ=0.2)
    baseline = mean(sars.q for sars in sars)
    -sum(sars) do sars
        π₀ = policy₀.π(sars.s)
        π′ = policy′.π(sars.s)
        a₀ = π₀[a_to_π_index(sars.a)]
        a′ = π′[a_to_π_index(sars.a)]
        min((a′ / a₀) * (sars.q - baseline), clip(a′ / a₀, 1 - ϵ, 1 + ϵ) * (sars.q - baseline))
    end / length(sars)
end

const π_optimizer = AMSGrad()

function train_policy!(policy, sars)
    policy₀ = deepcopy(policy)
    policy′ = policy
    for fit_iteration in Iterators.countfrom(1)
        pre_loss = π_loss(policy₀, policy′, sars)
        for _ in 1:10
            Flux.train!(sars -> π_loss(policy₀, policy′, sars), Flux.params(policy′.π), [(sample(sars, 100),)], π_optimizer)
        end
        post_loss = π_loss(policy₀, policy′, sars)
        if post_loss >= pre_loss; break end
    end
end

struct Policy <: AbstractPolicy
    π
end

Reinforce.action(policy::Policy, r, s, A) = sample(env.actions, Weights(policy.π(s)))

mutable struct SARS
    s
    a
    r
    q
    s′
    f
end

function run_episodes(n_episodes, policy; render_count=0, close_env=false)
    sars = []
    rewards = []
    for episode in 1:n_episodes
        reward = run_episode(env, policy) do (s, a, r, s′)
            push!(sars, SARS(s, a, r, 0, s′, finished(env)))
            if render_count >= episode; render(env) end
        end
        push!(rewards, reward)
    end
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

lastn(n, xs) = xs[max(1, end-n+1):end]

function run(policy=Policy(make_π_network()); stop_reward=200)
    try
        all_rewards = Float64[]
        mean_rewards = Tuple{Int64, Float64}[]
        batch_size = 100
        best_mean_reward = -Inf
        for iteration in Iterators.countfrom(1)
            @printf("%4d: batch size: %3d  ", iteration, batch_size)
            sars, rewards = run_episodes(batch_size, policy)
            append!(all_rewards, rewards)
            recent_rewards = lastn(100, all_rewards)
            push!(mean_rewards, (length(all_rewards), mean(recent_rewards)))
            if length(recent_rewards) == 100
                if mean(recent_rewards) > best_mean_reward
                    best_mean_reward = mean(recent_rewards)
                    batch_size = max(1, floor(batch_size * 0.8))
                else
                    batch_size += 1
                end
            end
            @printf("episodes: %5d  recent rewards: %7.2f  best reward: %7.2f\n", length(all_rewards), mean(recent_rewards), best_mean_reward)
            scatter(all_rewards, size=(1200, 800), markercolor=:blue, legend=false,
                    markersize=3, markeralpha=0.3,
                    markerstrokewidth=0, markerstrokealpha=0)
            plot!(mean_rewards, linecolor=:red,
                  linewidth=1, linealpha=0.5)
            display(scatter!(mean_rewards,
                             markercolor=:red, markershape=:vline,
                             markersize=11, markeralpha=0.2,
                             markerstrokewidth=0, markerstrokealpha=0))
            if mean(recent_rewards) >= stop_reward; break end
            train_policy!(policy, sars)
        end
    catch e
        if typeof(e) != InterruptException; rethrow() end
    end
    policy
end
