using DataStructures
using Gym
using Flux
using PyCall: PyArray
using StatsBase
using StatsBase: sample
using TensorBoardLogger
using Zygote

tb_log = TBLogger("tb_logs/sac")

Base.length(ds::DiscreteS) = ds.n
Base.iterate(ds::DiscreteS) = iterate(0:ds.n-1)
Base.iterate(ds::DiscreteS, state) = iterate(0:ds.n-1, state)
StatsBase.sample(ds::DiscreteS) = Gym.sample(ds)
StatsBase.sample(ds::DiscreteS, Weights) = StatsBase.sample(0:ds.n-1, Weights)

const env = GymEnv("CartPole-v1")

function polyak_average!(a, b, τ=0.01)
    for (pa, pb) in zip(params(a), params(b))
        pa .*= 1 - τ
        pa .+= pb * τ
    end
end

function flatten(x)
    reshape(x, length(x))
end

function valgrad(f, x...)
    val, back = pullback(f, x...)
    val, back(1)
end

function nonans(label)
    function (xs)
        @assert !any(isnan, xs) "nan at $label"
        xs
    end
end

function onehot(n, is)
    reduce(hcat, map(is) do i
        x = zeros(Float32, n)
        x[i] = 1f0
        x
    end)
end

struct SAR{S, A}
    s      :: S
    a      :: A
    r      :: Float32
    s′     :: S
    t      :: Int32
    failed :: Bool
    limit  :: Bool
end

abstract type AbstractPolicy end

function action end

function run_episode(step_f, env, policy)
    episode_reward = 0f0
    s = Gym.reset!(env)
    for t in Iterators.countfrom(1)
        a = action(policy, s, env.action_space)
        s′, r, failed, info = step!(env, a)
        episode_reward += r
        @assert t < env.gymenv._max_episode_steps
        limit = t == env.gymenv._max_episode_steps
        if limit
            failed = false
        end
        step_f(SAR(s, a, Float32(r), s′, Int32(t), failed, limit))
        if failed || limit
            break
        end
        s = s′
    end
    episode_reward
end

function make_π_network()
    Chain(
        nonans("π input"),
        Dense(4, 600, relu),
        Dense(600, 200, relu),
        Dense(200, 2, identity),
        softmax,
        nonans("π output"))
end

function make_q_network()
    Chain(
        nonans("q input"),
        Dense(4, 600, relu),
        Dense(600, 200, relu),
        Dense(200, 2, identity),
        nonans("q output"))
end

function Q(policy, s, a)
    # 0 is a valid action. Add 1 to move to Julia's 1-based indexing.
    min(
        policy.q1(s)[a+1],
        policy.q2(s)[a+1])
end

function Q_targ(policy, s, a)
    # 0 is a valid action. Add 1 to move to Julia's 1-based indexing.
    min(
        policy.q1_targ(s)[a+1],
        policy.q2_targ(s)[a+1])
end

const q1_opt = RMSProp(0.001)
const q2_opt = RMSProp(0.001)
const π_opt = RMSProp(0.001)

const γ = 0.9
const α = 0.3

function train!(policy, replay_buffer)
    training_transitions = sample(replay_buffer, 100)
    q_target::Vector{Float32} = collect(map(training_transitions) do t
        if t.failed
            @debug "failed" t.r
            t.r
        else
            a′_probs = policy.π(t.s′)
            sum(1:2) do a′_sample_index
                Zygote.@ignore @debug "" a′_sample_index
                # Subtract 1 to move from 1-based Julia indexing to 0-based Python / Gym indexing
                a′ = a′_sample_index - 1
                a′_prob = a′_probs[a′_sample_index]
                Q_term = Q_targ(policy, t.s′, a′)
                entropy_term = α * log(a′_prob)
                @debug "y terms" Q_term entropy_term
                a′_prob * (t.r + γ * (Q_term - entropy_term))
            end
        end
    end)
    s_stack::Matrix{Float32} = reduce(hcat, map(t -> t.s, training_transitions))
    # 0 is a valid action. Add 1 to move to Julia's 1-based indexing.
    a_stack::Matrix{Float32} = onehot(2, map(t -> t.a+1, training_transitions))
    @debug "pre q training" q_target
    @debug "" s_stack
    @debug "" a_stack
    for (qi, q_opt, q_network) in ((1, q1_opt, policy.q1), (2, q2_opt, policy.q2))
        q_params = params(q_network)
        loss, grads = valgrad(q_params) do
            q_out::Matrix{Float32} = q_network(s_stack)
            Zygote.@ignore @debug "q training" q_out
            Zygote.@ignore log_histogram(tb_log, "sac/q$(qi)_out", flatten(q_out))
            qav::Vector{Float32} = flatten(sum(q_out .* a_stack, dims=1))
            Zygote.@ignore @debug "" qav
            error::Vector{Float32} = (qav - q_target).^2
            mean(error)
        end
        @debug "q loss" loss
        log_value(tb_log, "sac/q$(qi)_loss", loss)
        Flux.Optimise.update!(q_opt, q_params, grads)
    end
    π_params = params(policy.π)
    loss, grads = valgrad(π_params) do
        -mean(training_transitions) do t
            a_probs = policy.π(t.s)
            Zygote.@ignore @debug "policy training" a_probs
            sum(1:2) do a_sample_index
                Zygote.@ignore @debug "" a_sample_index
                # Subtract 1 to move from 1-based Julia indexing to 0-based Python / Gym indexing
                a = a_sample_index - 1
                a_prob = a_probs[a_sample_index]
                Zygote.@ignore @debug "" a_prob
                Q_term = Q(policy, t.s, a)
                entropy_term = α * log(a_prob)
                Zygote.@ignore @debug "" Q_term entropy_term
                a_prob * (Q_term - entropy_term)
            end
        end
    end
    @debug "policy loss" loss
    log_value(tb_log, "sac/policy_loss", loss)
    Flux.Optimise.update!(π_opt, π_params, grads)
    polyak_average!(policy.q1_targ, policy.q1)
    polyak_average!(policy.q2_targ, policy.q2)
end

struct Policy{P, Q} <: AbstractPolicy
    π       :: P
    q1      :: Q
    q2      :: Q
    q1_targ :: Q
    q2_targ :: Q
end

function Policy()
    q1 = make_q_network()
    q2 = make_q_network()
    Policy(make_π_network(), q1, q2, deepcopy(q1), deepcopy(q2))
end

function action(policy::AbstractPolicy, s, A)
    network_out::Vector{Float32} = policy.π(s::Union{Vector{Float32}, PyArray{Float64, 1}})
    sample(A, Weights(network_out))
end

function main()
    replay_buffer = CircularBuffer{SAR}(10_000)
    recent_episode_rewards = CircularBuffer{Float32}(100)
    policy = Policy()
    for episode in 1:500
        log_value(tb_log, "sac/replay_buffer_length", length(replay_buffer))
        episode_reward = run_episode(env, policy) do sar
            push!(replay_buffer, sar)
            render(env)
        end
        push!(recent_episode_rewards, episode_reward)
        @info "" mean(recent_episode_rewards) length(replay_buffer)
        for _ in 1:10
            increment_step!(tb_log, 1)
            train!(policy, replay_buffer)
        end
    end
end

main()
