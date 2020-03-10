using BSON
using Distributed
using Flux
using OpenAIGym
using Plots
using Printf
using ProgressMeter
using Test: @test

pyplot()

const env = GymEnv(:LunarLander, :v2)

mutable struct SARS{S, A}
    s  :: S
    a  :: A
    r  :: Float32
    q  :: Union{Nothing, Float32}
    s′ :: S
    f  :: Bool
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

function test_fill_q()
    sars = [
        SARS(nothing, nothing, 1f0, nothing, nothing, false),
        SARS(nothing, nothing, 1f0, nothing, nothing, false),
        SARS(nothing, nothing, 1f0, nothing, nothing, true),
        SARS(nothing, nothing, 1f0, nothing, nothing, false),
        SARS(nothing, nothing, 1f0, nothing, nothing, true)]
    fill_q!(sars)
    @test sars[1].q == 3
    @test sars[2].q == 2
    @test sars[3].q == 1
    @test sars[4].q == 2
    @test sars[5].q == 1
end

struct Policy{Π, Q, V} <: AbstractPolicy
    π :: Π
    q :: Q
    v :: V
end

Policy() = Policy(make_π_network(), make_q_network(), make_v_network())

Reinforce.action(policy::Policy, r, s, A) = sample(env.actions.items, Weights(policy.π(s)))

function make_π_network(hidden_layer_size=32)
    Chain(
        Dense(length(env.state), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, length(env.actions), identity),
        softmax)
end

struct QNetwork{T}
    network :: T
end
Flux.@treelike QNetwork
(q::QNetwork)(s, a) = q.network(vcat(s, Flux.onehot(a, env.actions.items)))

function make_q_network(hidden_layer_size=32)
    QNetwork(Chain(
        Dense(length(env.state) + length(env.actions), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, 1, identity),
        first))
end

function make_v_network(hidden_layer_size=32)
    Chain(
        Dense(length(env.state), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, 1, identity),
        first)
end

Flux.@nograd a_to_π_index(a) = indexin(a, env.actions.items)[1]

clip(n, ϵ) = clamp(n, 1 - ϵ, 1 + ϵ)

function π_loss(policy₀, policy′, sars, ϵ=0.2)
    -sum(sars) do sars
        π₀ = policy₀.π(sars.s)
        π′ = policy′.π(sars.s)
        a₀ = π₀[a_to_π_index(sars.a)]
        a′ = π′[a_to_π_index(sars.a)]
        advantage = policy₀.q(sars.s, sars.a) - policy₀.v(sars.s)
        a_ratio = a′ / a₀
        min(
            a_ratio * advantage,
            clip(a_ratio, ϵ) * advantage)
    end / length(sars)
end

function q_loss(policy, sars)
    sum(sars) do sars
        (policy.q(sars.s, sars.a) - sars.q)^2
    end / length(sars)
end

function v_loss(policy, sars)
    sum(sars) do sars
        (policy.v(sars.s) - sars.q)^2
    end / length(sars)
end

function train_policy!(policy, sars)
    fill_q!(sars, discount_factor=0.99)
    v_optimizer = ADAM()
    q_optimizer = ADAM()
    π_optimizer = ADAM()
    @showprogress "Fitting v: " for fit_iteration in 1:1000
        Flux.train!(sars -> v_loss(policy, sars), Flux.params(policy.v), [(sample(sars, 100),)], v_optimizer)
    end
    @showprogress "Fitting q: " for fit_iteration in 1:1000
        Flux.train!(sars -> q_loss(policy, sars), Flux.params(policy.q), [(sample(sars, 100),)], q_optimizer)
    end
    policy₀ = deepcopy(policy)
    policy′ = policy
    @showprogress "Fitting π: " for fit_iteration in 1:1000
        Flux.train!(sars -> π_loss(policy₀, policy′, sars), Flux.params(policy′.π), [(sample(sars, 100),)], π_optimizer)
    end
end

function run_episodes(n_episodes, policy; render_count=0, kwargs...)
    sars = SARS{Vector{Float32}, Int8}[]
    rewards = Float32[]
    presults = if render_count < 1 && nprocs() > 1
        @showprogress "Env batch: " pmap(1:n_episodes) do _
            run_episodes′(1, policy, render_count=render_count, parallel=true, kwargs...)
        end
    else
        progress = Progress(n_episodes, "Env batch: ")
        map(1:n_episodes) do _
            r = run_episodes′(1, policy, render_count=render_count, parallel=false, kwargs...)
            next!(progress)
            r
        end
    end
    for (psars, prewards) in presults
        append!(sars, psars)
        append!(rewards, prewards)
    end
    sars, rewards
end

function run_episodes′(n_episodes, policy; render_count=0, close_env=false, parallel=false)
    if parallel
        run_env = GymEnv(env.name, env.ver)
    else
        run_env = env
    end
    sars = SARS{Vector{Float32}, Int8}[]
    rewards = Float32[]
    for episode in 1:n_episodes
        reward = run_episode(run_env, policy) do (s, a, r, s′)
            if parallel
                s = copy(s)
                a = copy(a)
                r = copy(r)
                s′ = copy(s′)
            end
            push!(sars, SARS{Vector{Float32}, Int8}(s, a, r, nothing, s′, finished(run_env)))
            if render_count >= episode; render(run_env) end
        end
        push!(rewards, reward)
    end
    sars, rewards
end

last(xs, n) = xs[max(1, end-n+1):end]

clear_lines(n) = print("\u1b[F\u1b[2K" ^ n)

function train_until_reward!(policy, stop_reward; fancy_output=false, save_policy=false)
    try
        print("\n" ^ 4)
        start_time = time()
        all_rewards = Float32[]
        summary_rewards = []
        means = Tuple{Float32, Float32}[]
        for training_iteration in Iterators.countfrom()
            batch_size = max(100, training_iteration)
            sars, rewards = run_episodes(batch_size, policy)
            if fancy_output; run_episodes(1, policy, render_count=1) end
            append!(all_rewards, rewards)
            recent_rewards = last(all_rewards, batch_size)
            push!(summary_rewards, summarystats(recent_rewards))
            push!(means, (length(all_rewards), summary_rewards[end].mean))
            if save_policy; bson(@sprintf("policy/policy_%03d.bson", training_iteration), policy=policy) end
            clear_lines(4)
            @printf("%3d: Time: %4.2f    Best Mean: %8.3f    Mean: %8.3f    IQR: %8.3f, %8.3f, %8.3f\n",
                    training_iteration, (time() - start_time) / 60^2, maximum(s.mean for s in summary_rewards),
                    summary_rewards[end].mean,
                    summary_rewards[end].q25, summary_rewards[end].median, summary_rewards[end].q75)
            if fancy_output
                scatter(all_rewards, size=(1200, 800), markercolor=:blue, legend=false,
                        markersize=3, markeralpha=0.3,
                        markerstrokewidth=0, markerstrokealpha=0)
                plot!(means, linecolor=:red,
                      linewidth=1, linealpha=0.5)
                display(scatter!(means,
                                 markercolor=:red, markershape=:vline,
                                 markersize=11, markeralpha=0.2,
                                 markerstrokewidth=0, markerstrokealpha=0))
            end
            if mean(recent_rewards) >= stop_reward; break end
            train_policy!(policy, sars)
        end
    catch e
        if typeof(e) != InterruptException; rethrow() end
    finally
        close(env)
    end
    policy
end

function run_tests()
    test_fill_q()
end

run_tests()
