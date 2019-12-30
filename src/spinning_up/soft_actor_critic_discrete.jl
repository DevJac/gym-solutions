module SACD

using DataStructures
using Flux
using OpenAIGym: AbstractPolicy, Reinforce, sample, Weights

entropy(ps) = ps' * -log.(max(1e-8, p) for p in ps)

Flux.@nograd onehot(x, set) = Flux.onehot(x, set.items)

function polyak_average!(network, other, p)
    for (network_param, other_param) in zip(Flux.params(network), Flux.params(other))
        network_param .= p * network_param + (1-p) * other_param
    end
end

function make_π_network(env, hidden_layer_size=32)
    Chain(
        Dense(length(env.state), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, length(env.actions), identity),
        softmax)
end

function π_loss(policy, sars)
    -sum(sars) do sars
        π_s = policy.π(sars.s)
        π_s' * policy.q(sars.s) + exp(policy.log_entropy_bonus[1]) * entropy(π_s)
    end / length(sars)
end

struct QNetwork
    env
    network1
    network2
end
Flux.@treelike QNetwork
(q::QNetwork)(s) = [q(s, a) for a in q.env.actions.items]
function (q::QNetwork)(s, a)
    network_input = vcat(s, onehot(a, q.env.actions))
    min(
        first(q.network1(network_input)),
        first(q.network2(network_input)))
end

function make_q_network(env, hidden_layer_size=32)
    network() = Chain(
        Dense(length(env.state) + length(env.actions), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, 1, identity))
    QNetwork(env, network(), network())
end

function q_loss(policy, sars)
    sum(sars) do sars
        (policy.q(sars.s, sars.a) - yq(policy, sars))^2
    end / length(sars)
end

function yq(policy, sars; discount_factor=0.99)
    if sars.f
        sars.r
    else
        sars.r + discount_factor * V(policy, sars.s′)
    end
end

function V(policy, s)
    π_s = policy.π(s)
    π_s' * policy.q_target(s) + exp(policy.log_entropy_bonus[1]) * entropy(π_s)
end

function eb_loss(policy, sars)
    sum(sars) do sars
        π_s = policy.π(sars.s)
        α = exp(policy.log_entropy_bonus[1])
        H = policy.target_entropy_bonus * (-log(1/length(π_s)))
        α * entropy(π_s) - sum(α * H * π_s)
    end / length(sars)
end

function train_policy!(policy, sars)
    append!(policy.replay_buffer, sars)
    q_optimizer = ADAM()
    π_optimizer = ADAM()
    eb_optimizer = ADAM()
    for fit_iteration in 1:100
        sars_sample = sample(policy.replay_buffer, 100)
        Flux.train!(sars -> q_loss(policy, sars), Flux.params(policy.q), [(sars_sample,)], q_optimizer)
        Flux.train!(sars -> π_loss(policy, sars), Flux.params(policy.π), [(sars_sample,)], π_optimizer)
        Flux.train!(sars -> eb_loss(policy, sars), Flux.params(policy.log_entropy_bonus), [(sars_sample,)], eb_optimizer)
        polyak_average!(policy.q_target, policy.q, 0.995)
    end
    println(exp(policy.log_entropy_bonus[1]))
end

struct Policy <: AbstractPolicy
    env  # The environment is not part of the policy, but the policy depends on the environment.
    π
    q
    q_target
    log_entropy_bonus
    target_entropy_bonus
    replay_buffer
    train_policy!
end

Reinforce.action(policy::Policy, r, s, A) = sample(policy.env.actions, Weights(policy.π(s)))

function make_default_policy(env)
    q = make_q_network(env)
    Policy(
        env,
        make_π_network(env),
        q,
        deepcopy(q),
        [0.0],
        0.5,
        CircularBuffer(1_000_000),
        train_policy!)
end

end  # end module SACD

include("base.jl")

function run()
    env = GymEnv(:CartPole, :v1)
    policy = SACD.make_default_policy(env)
    run_until_reward(policy, 495, batch_size=10, adaptive_batch_size=false)
end
