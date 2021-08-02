using DataStructures
using Flux
using LearnBase: DiscreteSet
using OpenAIGym
using PyCall
using Zygote

Base.iterate(set::DiscreteSet) = iterate(set.items)
Base.iterate(set::DiscreteSet, state) = iterate(set.items, state)
OpenAIGym.sample(set::DiscreteSet, weights) = sample(set.items, weights)

const env = GymEnv(:LunarLander, :v2)

function flatten(x)
    reshape(x, length(x))
end

function valgrad(f, x...)
    val, back = pullback(f, x...)
    val, back(1)
end

function nonans(label)
    function (xs)
        xs :: Union{Vector{Float32}, Matrix{Float32}, PyArray{Float32, 1}}
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


struct SARSD{S, A}
    s  :: S
    a  :: A
    r  :: Float32
    s′ :: S
    d  :: Bool
end

function make_π_network()
    Chain(
        nonans("π input"),
        Dense(8, 10),
        Dense(10, 4),
        softmax,
        nonans("π output"))
end

function make_q_network()
    Chain(
        nonans("q input"),
        Dense(8, 10),
        Dense(10, 4),
        softmax,
        nonans("q output"))
end

function Q(policy, s, a)
    # 0 is a valid action. Add 1 to move to Julia's 1-based indexing.
    min(
        policy.q1(s)[a+1],
        policy.q2(s)[a+1])
end

const q_opt = RMSProp(0.001)
const π_opt = RMSProp(0.001)

function train!(policy, replay_buffer)
    training_transitions = sample(replay_buffer, 100)
    q_target::Vector{Float32} = collect(map(training_transitions) do t
        γ = 0.99
        α = 0.01
        if t.d
            t.r
        else
            a′_probs = policy.π(t.s′)
            a′_sample_index = sample(1:4, Weights(a′_probs))
            # Subtract 1 to move from 1-based Julia indexing to 0-based Python / Gym indexing
            a′ = a′_sample_index - 1
            a′_prob = a′_probs[a′_sample_index]
            Q_term = Q(policy, t.s′, a′)
            entropy_term = α * log(a′_prob)
            @debug "y terms" Q_term entropy_term
            t.r + γ * (Q_term - entropy_term)
        end
    end)
    s_stack::Matrix{Float32} = reduce(hcat, map(t -> t.s, training_transitions))
    # 0 is a valid action. Add 1 to move to Julia's 1-based indexing.
    a_stack::Matrix{Float32} = onehot(4, map(t -> t.a+1, training_transitions))
    @debug "pre q training" q_target
    @debug "" s_stack
    @debug "" a_stack
    for q_network in (policy.q1, policy.q2)
        q_params = params(q_network)
        loss, grads = valgrad(q_params) do
            q_out::Matrix{Float32} = q_network(s_stack)
            Zygote.@ignore @debug "q training" q_out
            qav::Vector{Float32} = flatten(sum(q_out .* a_stack, dims=1))
            Zygote.@ignore @debug "" qav
            error::Vector{Float32} = (qav - q_target).^2
            mean(error)
        end
        Flux.Optimise.update!(q_opt, q_params, grads)
    end
    π_params = params(policy.π)
    loss, grads = valgrad(π_params) do
        -mean(training_transitions) do t
            α = 0.01
            a_probs = policy.π(t.s)
            Zygote.@ignore @debug "policy training" a_probs
            a_sample_index = Zygote.@ignore sample(1:4, Weights(a_probs))
            Zygote.@ignore @debug "" a_sample_index
            # Subtract 1 to move from 1-based Julia indexing to 0-based Python / Gym indexing
            a = a_sample_index - 1
            a_prob = a_probs[a_sample_index]
            Q_term = Q(policy, t.s, a)
            entropy_term = α * log(a_prob)
            Zygote.@ignore @debug "" Q_term entropy_term
            Q_term - entropy_term
        end
    end
    Flux.Optimise.update!(π_opt, π_params, grads)
end

struct Policy{P, Q} <: AbstractPolicy
    π :: P
    q1 :: Q
    q2 :: Q
end

Policy() = Policy(make_π_network(), make_q_network(), make_q_network())

function Reinforce.action(policy::AbstractPolicy, r, s, A)
    network_out::Vector{Float32} = policy.π(s::Union{Vector{Float32}, PyArray{Float32, 1}})
    sample(A, Weights(network_out))
end

function main()
    replay_buffer = CircularBuffer{SARSD}(10_000)
    policy = Policy()
    for episode in 1:20
        run_episode(env, policy) do (s, a, r, s′)
            push!(replay_buffer, SARSD(s, a, Float32(r), s′, finished(env)))
            render(env)
        end
        train!(policy, replay_buffer)
    end
end

main()
