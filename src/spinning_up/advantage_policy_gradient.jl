module AdvantagePG

using Flux
using OpenAIGym

struct SARS
    s
    a
    r
    q
    s′
    f
end

function fill_q(sars; discount_factor=1)
    sars′ = []
    q = 0
    for i in length(sars):-1:1
        q *= discount_factor
        if sars[i].f
            q = 0
        end
        q += sars[i].r
        push!(sars′, SARS(
            sars[i].s,
            sars[i].a,
            sars[i].r,
            q,
            sars[i].s′,
            sars[i].f))
    end
    sars′
end

function make_π_network(env, hidden_layer_size=32)
    Chain(
        Dense(length(env.state), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, length(env.actions), identity),
        softmax)
end

struct QNetwork
    env
    network
end
(q::QNetwork)(s, a) = q.network(vcat(s, Flux.onehot(a, q.env.actions)))
Flux.@treelike QNetwork

function make_q_network(env, hidden_layer_size=32)
    QNetwork(env, Chain(
        Dense(length(env.state) + length(env.actions), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, 1, identity),
        first))
end

a_to_π_index(env, a) = indexin(a, env.actions.items)[1]

v(policy, s) = sum(policy.π(s)[a_to_π_index(policy.env, a)] * policy.q(s, a) for a in policy.env.actions)

function π_loss(policy, sars)
    -sum(sars) do sars
        Φ = policy.q(sars.s, sars.a) - v(policy, sars.s)
        log(policy.π(sars.s)[a_to_π_index(policy.env, sars.a)]) * Φ
    end / length(filter(sars -> sars.f, sars))
end

function q_loss(policy, sars)
    sum(sars) do sars
        (policy.q(sars.s, sars.a) - sars.q)^2
    end / length(sars)
end

const π_optimizer = AMSGrad()
const q_optimizer = AMSGrad()

function train_policy!(policy, sars)
    sars = fill_q(sars)
    for fit_iteration in Iterators.countfrom(1)
        pre_loss = q_loss(policy, sars)
        for _ in 1:10
            Flux.train!(sars -> q_loss(policy, sars), Flux.params(policy.q), [(sample(sars, 100),)], q_optimizer)
        end
        post_loss = q_loss(policy, sars)
        if post_loss >= pre_loss; break end
    end
    Flux.train!(sars -> π_loss(policy, sars), Flux.params(policy.π), [(sars,)], π_optimizer)
end

struct Policy <: AbstractPolicy
    env  # The environment is not part of the policy, but the policy depends on the environment.
    π
    q
    train_policy!
end

Reinforce.action(policy::Policy, r, s, A) = sample(policy.env.actions, Weights(policy.π(s)))

function make_default_policy(env)
    Policy(env, make_π_network(env), make_q_network(env), train_policy!)
end

end  # end module AdvantagePG

include("base.jl")

function run()
    env = GymEnv(:CartPole, :v1)
    policy = SimplePG.make_default_policy(env)
    run_until_reward(policy, 495)
end
