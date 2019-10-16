module SimplePG

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

a_to_π_index(env, a) = indexin(a, env.actions.items)[1]

function π_loss(policy, sars)
    baseline = mean(sars.q for sars in sars)
    -sum(sars) do sars
        Φ = sars.q - baseline
        log(policy.π(sars.s)[a_to_π_index(policy.env, sars.a)]) * Φ
    end / length(filter(sars -> sars.f, sars))
end

const π_optimizer = ADAM(0.001)

function train_policy!(policy, sars)
    sars = fill_q(sars)
    Flux.train!(sars -> π_loss(policy, sars), Flux.params(policy.π), [(sars,)], π_optimizer)
end

struct Policy <: AbstractPolicy
    env  # The environment is not part of the policy, but the policy depends on the environment.
    π
    train_policy!
end

Reinforce.action(policy::Policy, r, s, A) = sample(policy.env.actions, Weights(policy.π(s)))

function make_default_policy(env)
    Policy(env, make_π_network(env), train_policy!)
end

end  # end module SimplePG

include("base.jl")

function run()
    env = GymEnv(:CartPole, :v1)
    policy = SimplePG.make_default_policy(env)
    run_until_reward(policy, 495)
end
